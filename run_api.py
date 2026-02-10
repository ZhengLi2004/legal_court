"""Run the lightweight FastAPI service for frontend integration."""

import uvicorn


def main() -> None:
    uvicorn.run("mas.api.server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
