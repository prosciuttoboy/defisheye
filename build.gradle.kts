plugins {
    java
    application
}

repositories {
    mavenCentral()
}

group = "dev.prosciutto"
version = "1.0-SNAPSHOT"
description = "defisheye"

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(21))
    }
}

application {
    mainClass.set("dev.prosciutto.defisheye.App")
    applicationDefaultJvmArgs = listOf("-Xss3m")
}

dependencies {
    implementation("org.bytedeco:javacv:1.5.9")
    implementation("org.bytedeco:openblas-platform:0.3.23-1.5.9")
    implementation("org.bytedeco:opencv-platform:4.7.0-1.5.9")
    implementation("org.bytedeco:ffmpeg-platform:6.0-1.5.9")
}
