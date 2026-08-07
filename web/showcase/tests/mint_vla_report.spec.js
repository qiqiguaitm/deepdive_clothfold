const { test, expect } = require("@playwright/test");

const reportUrl = "http://127.0.0.1:8765/reports/mint_vla_report/";

for (const viewport of [
  { name: "desktop", width: 1440, height: 1000 },
  { name: "mobile", width: 390, height: 844 },
]) {
  test(`${viewport.name} final report has no overflow or console errors`, async ({ page }) => {
    const errors = [];
    page.on("console", (message) => {
      if (message.type() === "error") errors.push(message.text());
    });

    await page.setViewportSize(viewport);
    await page.goto(reportUrl, { waitUntil: "networkidle" });

    const dimensions = await page.evaluate(() => ({
      clientWidth: document.documentElement.clientWidth,
      scrollWidth: document.documentElement.scrollWidth,
    }));
    expect(dimensions.scrollWidth).toBeLessThanOrEqual(dimensions.clientWidth + 1);
    expect(errors).toEqual([]);
    await expect(page.locator("#decision")).toContainText("0 pending · 0 running");
  });
}

test("final evidence and canonical attachments are synchronized", async ({ page, request }) => {
  await page.goto(reportUrl, { waitUntil: "networkidle" });

  await expect(page.locator("#summary")).toContainText("−8.58 pp");
  await expect(page.locator("#evidence")).toContainText("+1.94 pp");
  await expect(page.locator("#evidence")).toContainText("95% CI [−5.78,+12.75]");
  await expect(page.locator("#evidence")).toContainText("Ranking-size −13.0");
  await expect(page.locator("#interpretation")).toContainText("证据不支持");
  await expect(page.locator("#submission")).toContainText("等待外部发布");

  for (const asset of [
    "assets/mint_vla_paper.pdf",
    "assets/PAPER_TODO.md",
    "assets/PAPER_EVIDENCE_ARCHIVE_2026-08-01.md",
    "assets/PLAN_pi05_spatial_future_interface_2026-08-01.md",
  ]) {
    const response = await request.get(`${reportUrl}${asset}`);
    expect(response.ok()).toBeTruthy();
  }
});

test("showcase entry supports language switching and opens the final report", async ({ page }) => {
  await page.goto("http://127.0.0.1:8765/", { waitUntil: "networkidle" });
  await expect(page.locator(".report-entry")).toBeVisible();

  await page.getByRole("button", { name: "EN" }).click();
  await expect(page.locator(".report-entry h3")).toHaveText(
    "MINT-VLA: predictable futures do not ensure control utility",
  );

  await page.locator(".report-entry .report-link").click();
  await expect(page).toHaveURL(/\/reports\/mint_vla_report\/$/);
  await expect(page.locator("#summary")).toBeVisible();
});
