buildscript {
    repositories {
        google()
        maven { url = uri("https://repo1.maven.org/maven2") }
    }
    dependencies {
        classpath("com.android.tools.build:gradle:9.3.1")
        classpath("org.jetbrains.kotlin:kotlin-gradle-plugin:2.2.10")
    }
}

allprojects {
    repositories {
        google()
        maven { url = uri("https://repo1.maven.org/maven2") }
    }
}

tasks.register("clean").configure {
    delete("build")
}

