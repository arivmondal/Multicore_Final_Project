from __future__ import annotations

from pathlib import Path

from django.http import FileResponse, Http404
from django.shortcuts import render

from .forms import DemoForm
from .services.benchmark_runner import run_benchmark as run_benchmark_service
from .services.filter_runner import get_demo_images, get_sample_path, get_system_info, process_demo


def index(request):
    form = DemoForm()
    return render(
        request,
        "demoapp/index.html",
        {
            "form": form,
            "demo_images": get_demo_images(),
            "system_info": get_system_info(),
        },
    )


def run_demo(request):
    form = DemoForm(request.POST, request.FILES)
    context = {"form": form}
    if not form.is_valid():
        return render(request, "demoapp/partials/result_panel.html", context, status=400)

    try:
        context["result"] = process_demo(form.cleaned_data)
    except Exception as exc:
        context["fatal_error"] = str(exc)
        return render(request, "demoapp/partials/result_panel.html", context, status=500)
    return render(request, "demoapp/partials/result_panel.html", context)


def run_benchmark(request):
    form = DemoForm(request.POST, request.FILES)
    context = {"form": form}
    if not form.is_valid():
        return render(request, "demoapp/partials/benchmark_panel.html", context, status=400)

    try:
        context["benchmark"] = run_benchmark_service(form.cleaned_data)
    except Exception as exc:
        context["fatal_error"] = str(exc)
        return render(request, "demoapp/partials/benchmark_panel.html", context, status=500)
    return render(request, "demoapp/partials/benchmark_panel.html", context)


def sample_preview(request, sample_slug: str):
    try:
        sample_path = get_sample_path(sample_slug)
    except FileNotFoundError as exc:
        raise Http404(str(exc)) from exc
    if not Path(sample_path).exists():
        raise Http404("Sample image is missing.")
    return FileResponse(open(sample_path, "rb"))
