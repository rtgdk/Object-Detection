from __future__ import unicode_literals

from django.conf.urls import url

from app import views


urlpatterns = [
    url(r'^$', views.index, name='index'),
]