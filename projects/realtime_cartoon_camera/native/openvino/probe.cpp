// A small native runtime gate before adding CameraX/JNI. Input is already
// normalized NCHW float32, so no image codec or Python is needed on the device.
#include <openvino/openvino.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if (argc < 4 || argc > 5) {
            std::cerr << "Usage: cartoon_openvino_probe model.xml input.f32 output_prefix [repeats]\n";
            return 2;
        }
        const int repeats = argc == 5 ? std::stoi(argv[4]) : 10;
        if (repeats < 1 || repeats > 1000) throw std::runtime_error("repeats must be 1..1000");
        ov::Core core;
        auto compiled = core.compile_model(argv[1], "CPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
            ov::hint::inference_precision(ov::element::f32), ov::inference_num_threads(4));
        auto request = compiled.create_infer_request();
        auto input = request.get_input_tensor();
        if (input.get_element_type() != ov::element::f32 || input.get_shape().size() != 4 ||
            input.get_shape()[0] != 1 || input.get_shape()[1] != 3 || compiled.outputs().size() != 3)
            throw std::runtime_error("Expected batch=1 RGB float32 and three feature outputs");
        if (std::filesystem::file_size(argv[2]) != input.get_byte_size())
            throw std::runtime_error("Input byte count does not match the model shape");
        std::ifstream file(argv[2], std::ios::binary);
        if (!file.read(reinterpret_cast<char*>(input.data<float>()), static_cast<std::streamsize>(input.get_byte_size())))
            throw std::runtime_error("Cannot read normalized input");
        request.infer(); // warm up separately
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < repeats; ++i) request.infer();
        const double mean_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / repeats;
        std::cout << std::setprecision(8) << "{\"repeats\":" << repeats << ",\"mean_inference_ms\":" << mean_ms << ",\"outputs\":[";
        for (size_t i = 0; i < 3; ++i) {
            auto output = request.get_output_tensor(i);
            if (output.get_element_type() != ov::element::f32) throw std::runtime_error("Output must be float32");
            std::ofstream dump(std::string(argv[3]) + "." + std::to_string(i) + ".f32", std::ios::binary);
            if (!dump.write(reinterpret_cast<const char*>(output.data<float>()), static_cast<std::streamsize>(output.get_byte_size())))
                throw std::runtime_error("Cannot write feature tensor");
            if (i) std::cout << ',';
            std::cout << '[';
            const auto shape = output.get_shape();
            for (size_t j = 0; j < shape.size(); ++j) { if (j) std::cout << ','; std::cout << shape[j]; }
            std::cout << ']';
        }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
