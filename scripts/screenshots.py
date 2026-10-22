"""Browser check of the running app (system Edge via Playwright): clicks through every page, solves the
"Two triangles + bridge" example under Degree and Inverse degree, and saves screenshots to docs/screenshots/.

    streamlit run app/streamlit_app.py --server.port 8599 --server.headless true
    python scripts/screenshots.py [http://localhost:8599]
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8599"
OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)


def wait_idle(page, ms=1500):
    page.wait_for_selector('[data-testid="stApp"]', timeout=60000)
    page.wait_for_function("() => !document.querySelector('[data-testid=\"stStatusWidget\"] svg')", timeout=120000)
    page.wait_for_timeout(ms)


def main():
    errors = []
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True)
        page = b.new_page(viewport={"width": 1500, "height": 1300}, color_scheme="light")
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"{URL}/")  # default page (Solve) is served at the root
        wait_idle(page, 3000)
        for scheme, tag in (("Degree", "degree"), ("Inverse degree", "inverse")):
            page.get_by_role("radio", name=scheme, exact=True).first.click()
            wait_idle(page)
            page.get_by_role("button", name="Find total dominating set").click()
            page.wait_for_selector("text=Total weight W(S)", timeout=120000)
            wait_idle(page, 6000)                            # let the vis-network iframe draw
            page.screenshot(path=str(OUT / f"solve_two_triangles_{tag}.png"), full_page=True)
            body = page.inner_text("body")
            print(f"solve ({scheme}): 'Valid' shown = {'Valid' in body}")
        page.goto(f"{URL}/schemes")
        wait_idle(page, 2000)
        page.get_by_role("button", name="House: degree vs inverse").click()
        page.wait_for_selector("text=Hubs in S", timeout=180000)
        wait_idle(page, 4000)
        page.screenshot(path=str(OUT / "schemes_house.png"), full_page=True)
        for path in ("compare", "results", "how"):
            page.goto(f"{URL}/{path}")
            wait_idle(page, 2500)
            if path == "compare":
                page.get_by_role("button", name="Run all methods").click()
                page.wait_for_selector("text=Exact ILP", timeout=180000)
                wait_idle(page, 2500)
            page.screenshot(path=str(OUT / f"{path}.png"), full_page=True)
            exc = page.locator('[data-testid="stException"]').count()
            print(f"{path}: exceptions on page = {exc}")
        b.close()
    print("page errors:", errors or "none")
    print("screenshots in", OUT)


if __name__ == "__main__":
    main()
