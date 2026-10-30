$env:JAVA_HOME = 'D:\IDEAS\jdk\jdk-21.0.8'
$env:ANDROID_HOME = 'B:\tmp\realtime-android-sdk'
$env:ANDROID_SDK_ROOT = $env:ANDROID_HOME
$env:NDK_HOME = "$env:ANDROID_HOME\ndk\28.2.13676358"
$env:GRADLE_USER_HOME = 'B:\tmp\realtime-gradle'
$env:Path = "C:\Users\Aogo\.cargo\bin;$env:JAVA_HOME\bin;$env:ANDROID_HOME\platform-tools;" + $env:Path
