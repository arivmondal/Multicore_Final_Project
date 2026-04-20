from __future__ import annotations

from django.urls import path

from . import views


app_name = "demoapp"

urlpatterns = [
    path("", views.index, name="index"),
    path("run/", views.run_demo, name="run_demo"),
    path("benchmark/", views.run_benchmark, name="run_benchmark"),
    path("samples/<slug:sample_slug>/", views.sample_preview, name="sample_preview"),
]
