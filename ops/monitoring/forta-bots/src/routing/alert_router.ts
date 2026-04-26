/**
 * Alert Router
 *
 * Routes Forta findings to the configured war-room channels per
 * Exploit Response Playbook §3 communication protocol.
 *
 * Channels:
 * - Discord war-room webhook (primary, fast)
 * - Optional: PagerDuty (post-mainnet, paged on-call)
 * - Optional: Email backup
 *
 * Severity mapping:
 * - Forta CRITICAL → Playbook P0 (active drain / key compromise / known exploit executing)
 * - Forta HIGH     → Playbook P1 (confirmed vulnerability, exploit feasible)
 * - Forta MEDIUM   → Playbook P2 (suspected anomaly; investigation needed)
 * - Forta LOW      → Playbook P3 (hardening opportunity)
 * - Forta INFO     → no paging; logged only
 *
 * Webhook URLs are runtime-configured via env vars; never committed.
 */

import { Finding, FindingSeverity } from "forta-agent";

export type PagerSeverity = "P0" | "P1" | "P2" | "P3" | "INFO";

export interface RouterConfig {
  discordWebhookUrl?: string;
  pagerDutyIntegrationKey?: string;
  emailRecipientList?: string[];
  dryRun?: boolean;  // log only, do not POST
}

export class AlertRouter {
  private config: RouterConfig;

  constructor(config: RouterConfig = {}) {
    this.config = {
      discordWebhookUrl: process.env.PRSM_DISCORD_WEBHOOK_URL,
      pagerDutyIntegrationKey: process.env.PRSM_PAGERDUTY_KEY,
      emailRecipientList: (process.env.PRSM_ALERT_EMAILS || "")
        .split(",")
        .map((e) => e.trim())
        .filter(Boolean),
      dryRun: process.env.PRSM_ALERT_DRY_RUN === "1",
      ...config,
    };
  }

  /**
   * Route a finding to all configured channels.
   * Returns the playbook severity classification for downstream handling.
   */
  async route(finding: Finding): Promise<PagerSeverity> {
    const severity = this.mapSeverity(finding.severity);

    if (this.config.dryRun) {
      console.error(
        `[DRY RUN] ${severity} ${finding.alertId} ${finding.name}: ${finding.description}`
      );
      return severity;
    }

    // Best-effort fan-out — failures in one channel do not block others
    const tasks: Promise<void>[] = [];
    if (this.config.discordWebhookUrl && severity !== "INFO") {
      tasks.push(this.sendDiscord(finding, severity));
    }
    if (this.config.pagerDutyIntegrationKey && (severity === "P0" || severity === "P1")) {
      tasks.push(this.sendPagerDuty(finding, severity));
    }
    if (this.config.emailRecipientList && this.config.emailRecipientList.length > 0 && severity === "P0") {
      // Email only on P0 to avoid noise
      tasks.push(this.sendEmail(finding, severity));
    }

    await Promise.allSettled(tasks);
    return severity;
  }

  /**
   * Map Forta finding severity to Playbook P0-P3 classification.
   */
  private mapSeverity(fortaSeverity: FindingSeverity): PagerSeverity {
    switch (fortaSeverity) {
      case FindingSeverity.Critical:
        return "P0";
      case FindingSeverity.High:
        return "P1";
      case FindingSeverity.Medium:
        return "P2";
      case FindingSeverity.Low:
        return "P3";
      case FindingSeverity.Info:
      default:
        return "INFO";
    }
  }

  /**
   * Discord webhook payload — embed-formatted finding for #war-room-active channel.
   */
  private async sendDiscord(finding: Finding, severity: PagerSeverity): Promise<void> {
    if (!this.config.discordWebhookUrl) return;

    const colorMap: Record<PagerSeverity, number> = {
      P0: 0xff0000,  // red
      P1: 0xff8c00,  // dark orange
      P2: 0xffd700,  // gold
      P3: 0x4682b4,  // steel blue
      INFO: 0x808080, // gray
    };

    const payload = {
      embeds: [
        {
          title: `[${severity}] ${finding.name}`,
          description: finding.description,
          color: colorMap[severity],
          fields: [
            { name: "Alert ID", value: finding.alertId || "n/a", inline: true },
            { name: "Protocol", value: finding.protocol || "PRSM", inline: true },
            { name: "Severity", value: severity, inline: true },
            ...Object.entries(finding.metadata || {}).slice(0, 10).map(([k, v]) => ({
              name: k,
              value: String(v).slice(0, 1024),
              inline: false,
            })),
          ],
          footer: { text: "PRSM Forta Monitoring · Playbook §3" },
          timestamp: new Date().toISOString(),
        },
      ],
    };

    try {
      const res = await fetch(this.config.discordWebhookUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        console.error(`Discord webhook failed: ${res.status} ${res.statusText}`);
      }
    } catch (e) {
      console.error(`Discord webhook error:`, e);
    }
  }

  /**
   * PagerDuty Events API v2 — pages on-call council member for P0/P1.
   */
  private async sendPagerDuty(finding: Finding, severity: PagerSeverity): Promise<void> {
    if (!this.config.pagerDutyIntegrationKey) return;

    const payload = {
      routing_key: this.config.pagerDutyIntegrationKey,
      event_action: "trigger",
      payload: {
        summary: `[${severity}] ${finding.name}`,
        severity: severity === "P0" ? "critical" : "error",
        source: "prsm-forta-bot",
        custom_details: {
          alertId: finding.alertId,
          description: finding.description,
          protocol: finding.protocol,
          metadata: finding.metadata,
        },
      },
    };

    try {
      const res = await fetch("https://events.pagerduty.com/v2/enqueue", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        console.error(`PagerDuty failed: ${res.status} ${res.statusText}`);
      }
    } catch (e) {
      console.error(`PagerDuty error:`, e);
    }
  }

  /**
   * Email fallback — placeholder. Real implementation depends on
   * Foundation's email provider choice (SES / SendGrid / Postmark).
   */
  private async sendEmail(finding: Finding, severity: PagerSeverity): Promise<void> {
    // Placeholder — integration with email provider deferred until Foundation
    // operations contract decisions complete (Phase 5 vendor selection).
    console.error(
      `[email-stub] ${severity} ${finding.name} would email: ${this.config.emailRecipientList?.join(", ")}`
    );
  }
}

/**
 * Singleton router instance for use across detectors.
 */
let _router: AlertRouter | null = null;

export function getAlertRouter(): AlertRouter {
  if (!_router) {
    _router = new AlertRouter();
  }
  return _router;
}
