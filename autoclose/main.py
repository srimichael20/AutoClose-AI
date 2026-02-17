"""AutoClose AI - Main entry point."""

import uvicorn

from autoclose.api.app import create_app
from autoclose.config import get_settings


def main() -> None:
    """Run the FastAPI server."""
    settings = get_settings()
    app = create_app()
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
    )


if __name__ == "__main__":
    main()
