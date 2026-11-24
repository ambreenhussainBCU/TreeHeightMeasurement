[app]
title = Tree Divider
package.name = treedivider
package.domain = org.example
source.dir = .
source.include_exts = py,png,jpg,kv
source.exclude_dirs = tests,bin,.buildozer
version = 1.0
requirements = python3,kivy==2.3.1,pillow,plyer
orientation = portrait
fullscreen = 0
android.permissions = CAMERA, READ_MEDIA_IMAGES, READ_EXTERNAL_STORAGE, WRITE_EXTERNAL_STORAGE
android.api = 34
android.minapi = 24
android.archs = arm64-v8a, armeabi-v7a

[buildozer]
log_level = 2
