/**
 * Global teardown for Jest tests
 */

export default async function globalTeardown(): Promise<void> {
  console.log('\n✅ PRSM SDK tests completed!');
}