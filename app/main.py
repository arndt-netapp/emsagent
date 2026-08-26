from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api import routes_analysis, routes_candidates, routes_clusters, routes_files, routes_findings
from app.config import REPO_ROOT
from app.db.session import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="ONTAP EMS Log Agent", lifespan=lifespan)

app.include_router(routes_files.router)
app.include_router(routes_analysis.router)
app.include_router(routes_candidates.router)
app.include_router(routes_findings.router)
app.include_router(routes_clusters.router)

class NoCacheStaticFiles(StaticFiles):
    """Serve the frontend with `no-cache`, so the browser revalidates instead of
    reusing a stale copy.

    This is not a micro-optimization worry, it is a correctness one: the HTML
    and the JS that drives it are deployed as separate files, and a browser
    holding an old script against new markup produces silent, confusing
    breakage. That happened for real — a cached files.js still referencing a
    removed element threw on `getElementById(...).addEventListener`, which
    aborted the script before the cluster-fetch form's submit handler was
    attached, so the form fell back to a native GET and serialized the cluster
    password into the URL (and the server access log).

    `no-cache` means "revalidate before reusing", not "never cache" — responses
    are still 304'd when unchanged, so the cost is one conditional request per
    asset.
    """

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


STATIC_DIR = REPO_ROOT / "web" / "static"
app.mount("/", NoCacheStaticFiles(directory=str(STATIC_DIR), html=True), name="static")
