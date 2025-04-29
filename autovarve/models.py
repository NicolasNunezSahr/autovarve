from django.db import models
from django.utils import timezone


class PipeRun(models.Model):
    mode = models.CharField(max_length=50)  # RGB or GRAY
    scale_pixel_value_max = models.FloatField(null=True, blank=True)
    crop_left = models.IntegerField(null=True, blank=True)
    crop_right = models.IntegerField(null=True, blank=True)
    crop_top = models.IntegerField(null=True, blank=True)
    crop_bottom = models.IntegerField(null=True, blank=True)
    kernel_size_horizontal = models.IntegerField(null=True, blank=True)
    kernel_size_vertical = models.IntegerField(null=True, blank=True)
    kernel_function_horizontal = models.CharField(max_length=50, null=True, blank=True)
    kernel_function_vertical = models.CharField(max_length=50, null=True, blank=True)
    pixel_change_threshold = models.FloatField(null=True, blank=True)
    vertical_or_aggregation_size = models.IntegerField(null=True, blank=True, default=0)
    column_fraction_threshold = models.FloatField(null=True, blank=True, default=0)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"PipeRun {self.id}"


class CoreColumn(models.Model):
    pipe_run = models.ForeignKey(PipeRun, on_delete=models.CASCADE)
    column_order = models.IntegerField(null=True, blank=True)
    column_width = models.IntegerField(null=True, blank=True)
    pixel_start = models.IntegerField(null=True, blank=True)
    pixel_end = models.IntegerField(null=True, blank=True)
    varve_count = models.IntegerField(null=True, blank=True)
