"""
URL configuration for storia project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from core import views


urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/hello/', views.hello_world_view),  # Map the function directly
    path('upsert-tweets/', views.upsert_tweets, name='upsert_tweets'),
    path('answer_query/', views.answer_query, name='answer_query')
    path('get_top_topics/', views.get_top_topics, name='get_top_topics'),
]

