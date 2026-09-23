plugins {
    alias(libs.plugins.android.application)
}

val qnnSdkRoot: String? = providers.gradleProperty("qnn.sdk.dir").orNull
    ?: providers.environmentVariable("QNN_SDK_ROOT").orNull
    ?: providers.fileContents(rootProject.layout.projectDirectory.file("local.properties"))
        .asText.orNull
        ?.lineSequence()
        ?.map { it.trim() }
        ?.firstOrNull { it.startsWith("qnn.sdk.dir=") }
        ?.substringAfter("=")
        ?.trim()

android {
    namespace = "com.example.yolovulkanmobile"
    ndkVersion = "28.2.13676358"
    compileSdk {
        version = release(37)
    }

    defaultConfig {
        applicationId = "com.example.yolovulkanmobile"
        minSdk = 24
        targetSdk = 37
        versionCode = 1
        versionName = "1.0"

        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a", "x86_64")
        }
        externalNativeBuild {
            cmake {
                arguments += "-DANDROID_STL=c++_shared"
                if (!qnnSdkRoot.isNullOrEmpty()) arguments += "-DQNN_SDK_ROOT=$qnnSdkRoot"
                cppFlags += "-std=c++17"
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    buildFeatures {
        viewBinding = true
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    androidResources {
        noCompress += listOf("param", "bin")
    }
    packaging {
        jniLibs {
            useLegacyPackaging = true
            excludes += listOf("**/libQnnDsp*.so", "**/libQnnGpu.so", "**/libQnnHtpPrepare.so")
        }
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.activity.ktx)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.camera.core)
    implementation(libs.androidx.camera.camera2)
    implementation(libs.androidx.camera.lifecycle)
    implementation(libs.androidx.camera.view)
    implementation(libs.qnn.runtime)
    testImplementation(libs.junit)
}
