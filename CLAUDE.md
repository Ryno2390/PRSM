## Testing Methodology

- When performing tests do not deprecate to simpler testing methods - work through the problem and if, need be, download packages to aid in your debugging.

## GitHub Workflow

- When pushing to GitHub, use SSH (key: SHA256:7X+8TXzSjBE88FP41JQClH1uHWjLbKP/LEAVTKSIapQ) commit and push changes.
- Before pushing to GitHub, always audit the repo to make sure that only essential files are located in the root directory, all other files are saved to logical subfolders, and make sure that no unnecessary test or irrelevant to-do/roadmap files are in the repo. Always make sure the repo is clean and organized enough for an external audit by either investors or developers.

## Development Best Practices

- Whenever possible/appropriate edit existing files rather than creating new ones. Otherwise, you may create a daisy chain of dependencies that turn into knots when trying to debug.