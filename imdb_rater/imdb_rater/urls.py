"""imdb_rater URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url, include
from django.contrib import admin
from django.contrib.auth import views as auth_views
from src import views as core_views
from django.conf import settings
from django.views.static import serve

urlpatterns = [
    url(r'^$', core_views.home, name='home'),
    url(r'^home', core_views.post_rating, name='post_rating'),
    url(r'^trust', core_views.trust, name='trust'),
    url(r'^post_trust', core_views.post_trust, name='post_trust'),
    url(r'^display', core_views.display, name='display'),
    url(r'^user_reaction', core_views.user_reaction, name='user_reaction'),
    url(r'^algo', core_views.algo, name='algo'),
    url(r'^login/$', auth_views.login, name='login'),
    url(r'^logout/$', auth_views.logout, name='logout'),
    url(r'^oauth/', include('social_django.urls', namespace='social')),
    url(r'^admin/', admin.site.urls),
    url(r'^media/(?P<path>.*)$', serve,{'document_root': settings.MEDIA_ROOT}),
]
