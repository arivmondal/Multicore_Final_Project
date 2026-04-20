from __future__ import annotations

from django import forms

from .services.filter_runner import get_demo_image_choices
from .services.kernel_factory import FILTER_CHOICES


RUN_MODE_CHOICES = [
    ("both", "Baseline + Low-Rank"),
    ("baseline", "Full 2D Baseline Only"),
    ("low_rank", "Low-Rank Only"),
]


class DemoForm(forms.Form):
    input_image = forms.FileField(required=False)
    sample_image = forms.ChoiceField(required=False)
    filter_type = forms.ChoiceField(choices=FILTER_CHOICES, initial="gaussian")
    kernel_size = forms.IntegerField(min_value=3, initial=31)
    rank = forms.IntegerField(min_value=1, initial=4)
    run_mode = forms.ChoiceField(choices=RUN_MODE_CHOICES, initial="both")
    sigma = forms.FloatField(required=False, min_value=0.0, initial=5.0)
    disk_radius = forms.FloatField(required=False, min_value=0.0, initial=8.0)
    motion_thickness = forms.IntegerField(required=False, min_value=1, initial=1)
    benchmark_ranks = forms.CharField(required=False, initial="1,2,4,8,16")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["sample_image"].choices = [("", "Choose a demo image")] + get_demo_image_choices()

    def clean(self):
        cleaned = super().clean()
        if not cleaned.get("input_image") and not cleaned.get("sample_image"):
            raise forms.ValidationError("Upload an image or choose one of the included demo images.")

        kernel_size = cleaned.get("kernel_size")
        if kernel_size and kernel_size % 2 == 0:
            self.add_error("kernel_size", "Kernel size must be a positive odd integer.")

        return cleaned
