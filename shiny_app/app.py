"""RNAi screen Shiny dashboard entry point."""

from shiny import App
from app_components.server import server
from app_components.ui import app_ui

app = App(app_ui, server)
