const { test, expect } = require("@playwright/test");

const reportUrl = "http://127.0.0.1:8765/reports/mint_vla_report/";

for (const viewport of [
  { name: "desktop", width: 1440, height: 1000 },
  { name: "mobile", width: 390, height: 844 },
]) {
  test(`${viewport.name} report has no overflow or console errors`, async ({ page }) => {
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
  });
}

test("experiment filters and attachments work", async ({ page, request }) => {
  await page.goto(reportUrl, { waitUntil: "networkidle" });

  await page.getByRole("button", { name: "进行中" }).click();
  await expect(page.locator(".todo-item:visible")).toHaveCount(3);
  await page.getByRole("button", { name: "等待门槛" }).click();
  await expect(page.locator(".todo-item:visible")).toHaveCount(0);
  await page.getByRole("button", { name: "后续计划" }).click();
  await expect(page.locator(".todo-item:visible")).toHaveCount(3);

  await expect(page.locator("#evidence")).toContainText("所有 pooled Holm-adjusted p=1");
  await expect(page.locator("#evidence")).toContainText("67,265 MiB");
  await expect(page.locator("#evidence")).toContainText("0.8134 vs persistence 0.7479");
  await expect(page.locator("#evidence")).toContainText("89.58% vs 89.23%");
  await expect(page.locator("#evidence")).toContainText("Correct pooled 胜出 0/9");
  await expect(page.locator("#evidence")).toContainText("95% CI [+0.0391,+0.1110]");
  await expect(page.locator("#gates")).toContainText("Closed · adverse");

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

test("showcase entry supports language switching and opens the report", async ({ page }) => {
  await page.goto("http://127.0.0.1:8765/", { waitUntil: "networkidle" });
  await expect(page.locator(".report-entry")).toBeVisible();

  await page.getByRole("button", { name: "EN" }).click();
  await expect(page.locator(".report-entry h3")).toHaveText(
    "MINT-VLA: future milestones for VLA policies",
  );

  await page.locator(".report-entry .report-link").click();
  await expect(page).toHaveURL(/\/reports\/mint_vla_report\/$/);
  await expect(page.locator("#summary")).toBeVisible();
});
