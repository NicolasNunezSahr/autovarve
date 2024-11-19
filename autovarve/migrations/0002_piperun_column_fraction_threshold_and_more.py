# Generated by Django 5.1.3 on 2024-11-18 01:28

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('autovarve', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='piperun',
            name='column_fraction_threshold',
            field=models.FloatField(blank=True, default=0, null=True),
        ),
        migrations.AddField(
            model_name='piperun',
            name='vertical_or_aggregation_size',
            field=models.IntegerField(blank=True, default=0, null=True),
        ),
    ]
