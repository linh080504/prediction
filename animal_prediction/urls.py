"""animal_prediction URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
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
from animal import views
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    path('admin/', admin.site.urls),
    path('predict/', views.predict, name='predict'),
    path('home', views.home, name='home'),
    path('login', views.login_page, name='login'),
    path('', views.login_page, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('signup', views.signup_page, name='signup'),
    path('process-video-frame/', views.process_video_frame, name='process_video_frame'),
    path('profile/', views.profile, name='profile'),

    path('admin_page/users/', views.list_users, name='list_users'),
    path('users/edit/<int:user_id>/', views.edit_user, name='edit_user'),
    path('users/delete/<int:user_id>/', views.delete_user, name='delete_user'),
    path('admin_page/predictions/', views.list_predictions, name='list_predictions'),
    path('admin_page/', views.admin_page, name='admin_page'),
    path('admin_page/users/add/', views.add_user, name='add_user'),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
