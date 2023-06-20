// #include <Eigen/Core>
#include <torch/script.h>
#include "openvino/openvino.hpp"
#include <opencv2/opencv.hpp>
#include <ngraph/type/element_type.hpp>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>


#define CLOCK std::chrono::steady_clock
#define CLOCK_CAST std::chrono::duration_cast<std::chrono::microseconds>

struct Detection
{
    int class_id{0};
    std::string className{};
    float confidence{0.0};
    cv::Scalar color{};
    cv::Rect box{};
};

struct Detection_mask
{
    int class_id{0};
    std::string className{};
    float confidence{0.0};
    cv::Scalar color{};
    cv::Rect box{};
    std::vector<float> mask;
};


std::vector<Detection_mask> parsing_boxes(cv::Mat image, auto out_data1, auto box_shape1, auto box_type, float modelScoreThreshold, float modelNMSThreshold, int number_classes, std::string &name_classes) {

    std::time_t time0;
    std::time_t time1;
    std::time_t time2;
    std::time_t time3;
    // auto time4;

    cv::Size modelShape(640,640);
    bool yolov8;
    auto rows = box_shape1[1];
    auto dimensions = box_shape1[2];
    auto maxim_data = 0;

    cv::Size s(dimensions,rows);
    cv::Mat output_boxes(s, CV_32FC1);

    for(int row = 0; row < rows; row++) {
        for(int dimension = 0; dimension < dimensions; dimension++) {
    
            output_boxes.at<float>(row,dimension) = out_data1[dimensions*row + dimension];
    
        }
    }

    if (dimensions > rows) // Check if the shape[2] is more than shape[1] (yolov8)
    {
        yolov8 = true;
        rows = box_shape1[2];
        dimensions = box_shape1[1];
        std::cout << "output_boxes sizes: dims = " << output_boxes.size() << std::endl;
        output_boxes = output_boxes.reshape(1, dimensions);
        std::cout << "output_boxes sizes: dims = " << output_boxes.size() << std::endl;
        cv::transpose(output_boxes, output_boxes);
        std::cout << "output_boxes sizes: dims = " << output_boxes.size() << std::endl;
    }

    float *data = (float *)output_boxes.data;
    float x_factor = image.cols / modelShape.width;
    float y_factor = image.rows / modelShape.height;
    std::cout << "x_factor " << x_factor << " and y_factor " << y_factor << std::endl;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<float>> detection_masks;
    std::vector<float> mass;

    for (int i = 0; i < rows; ++i)
    {
        if (yolov8)
        {
            float *classes_scores = data+4;
            double maxClassScore;
            cv::Mat scores(1, number_classes, CV_32FC1, classes_scores);
            cv::Point class_id;
            
            cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
            
            if (maxClassScore > modelScoreThreshold)
            {
                confidences.push_back(maxClassScore);
                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
                for (int index=0; index < 32; index++) {
                    mass.push_back(*(data + 4 + number_classes + index));
                }
                detection_masks.push_back(mass);
                
            }
        }
        data += dimensions;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, modelScoreThreshold, modelNMSThreshold, nms_result);
    
    std::vector<Detection> detections{};
    std::vector<Detection_mask> detections_mask{};

    for (unsigned long i = 0; i < nms_result.size(); ++i)
    {
        int idx = nms_result[i];

        Detection result;
        Detection_mask result_mask;

        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];

        result_mask.class_id = class_ids[idx];
        result_mask.confidence = confidences[idx];

        result.color = cv::Scalar(255, 0, 0);
        result_mask.color = cv::Scalar(255, 0, 0);

        result.className = name_classes;
        result.box = boxes[idx];

        result_mask.className = name_classes;
        result_mask.box = boxes[idx];

        result_mask.mask = detection_masks[idx];

        detections.push_back(result);
        detections_mask.push_back(result_mask);
    }
    return detections_mask;
}

int main()
{
    std::chrono::steady_clock::time_point end_global;
    std::chrono::steady_clock::time_point begin_global;
    std::chrono::steady_clock::time_point begin_for_average;
    std::chrono::steady_clock::time_point end_for_average;
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;    
    // auto xml = "/home/ss21mipt/Documents/starkit/DIPLOMA/YOEO/config/IR&onnx_for_416_Petr_1/yoeo.xml";
    // auto xml = "/home/ss21mipt/Documents/starkit/DIPLOMA/to_rhoban/weights/Feds_yolov8_2_openvino/best.xml";
    // auto xml = "/home/ss21mipt/DIPLOMA/weights/best_openvino_model/best.xml";
    // auto xml = "/home/ss21mipt/DIPLOMA/weights/yolov8n-seg_openvino_22_opset14/Feds.xml";
    // auto xml = "/home/ss21mipt/DIPLOMA/weights/yolov8-seg_Feds_ov-2022.3.0_opset-14/yolov8-seg_Feds_12.xml";
    auto xml = "/home/ss21mipt/DIPLOMA/weights/yolov8/yolov8n-seg_openvino_22_opset14/Feds.xml";
    auto png = "/home/ss21mipt/Pictures/photo_2023-03-28_12-46-25.jpg";
    ov::Core core;

    std::shared_ptr<ov::Model> net = core.read_model(xml);    // net = ie.ReadNetwork(model_path);
    ov::preprocess::PrePostProcessor ppp(net);        
    ov::preprocess::InputInfo& input = ppp.input(0);            // inputs = net.getInputsInfo(); 
    auto input_shape = net->input(0).get_partial_shape();
    auto output1_shape = net->output(0).get_partial_shape();
    // std::cout << net->input(0).get_partial_shape() << output1_shape << std::endl;

    input.tensor().set_element_type(ov::element::f32);   // NAPISAL PODRYGOMY (NE NHWC) почему здесь другая нумерация, как это может быть правильно написанным вариантом?
    input.model().set_layout("NHWC");
    
    net = ppp.build();
    
    ov::CompiledModel compiled_model = core.compile_model(net, "CPU");    // auto executable_network = ie.LoadNetwork(net, device);
    ov::InferRequest infer_request = compiled_model.create_infer_request();    // infer_request = executable_network.CreateInferRequest();
 
    ov::Tensor m_inputData = infer_request.get_input_tensor(0);    // m_inputData = infer_request.GetBlob(inputsName);
    
    auto m_numChannels = m_inputData.get_shape()[1];
    auto m_inputW = m_inputData.get_shape()[3];
    auto m_inputH = m_inputData.get_shape()[2];
    auto m_imageSize = m_inputH * m_inputW;
    auto data1 = m_inputData.data<float_t>();
    
    // cv::VideoCapture cap(0);
    cv::VideoCapture cap("/home/ss21mipt/DIPLOMA/test_data/sahr3/video.avi");
    cv::Mat image;
    cv::Mat segments;
    cv::Mat mask;
    cv::Mat* img;
    cv::Mat test_mask;
    cv::Size modelShape(640,640);

    bool yolov8;
    auto modelScoreThreshold = 0.35;
    auto modelNMSThreshold = 0.5;
    int number_classes = 1;
    int numcycles = 0;
    std::string name_classes = "g";
    auto time_preproc= 0;
    auto t_preproc = 0;
    auto t_preproc_max = 0;
    auto t_preproc_min = 100;
    auto time_infer = 0;
    auto t_infer = 0;
    auto t_infer_max = 0;
    auto t_infer_min = 100;
    auto time_postproc = 0;
    auto t_postproc = 0;
    auto t_postproc_max = 0;
    auto t_postproc_min = 100;
    
    begin_for_average = CLOCK::now();
    while (numcycles < 500){//cap.isOpened()){
        begin_global = CLOCK::now();

        begin = CLOCK::now();
        cap >> image;
        // image = cv::imread("/home/ss21mipt/DIPLOMA/IoU_tool/ground_t/frame111.jpg");
        // image = cv::imread(png);

        if (image.empty() || !image.data) {
            return false;
        }
        cv::Size scale(640, 640);  
        cv::resize(image, image, scale);    
        // end = CLOCK::now();
        // std::cout << "FPS #1 IMREAD AND PREPROC " <<  (CLOCK_CAST(end - begin).count() / 1000.0) << std::endl;
        
        // begin = CLOCK::now();                       // FILLING DATA1
        for (size_t row = 0; row < m_inputH; row++) {
            for (size_t col = 0; col < m_inputW; col++) {
                for (size_t ch = 0; ch < m_numChannels; ch++) {

                    data1[m_imageSize * ch + row * m_inputW + col] = float(image.at<cv::Vec3b>(row, col)[ch] / 255.0);

                }
            }
        }

        // std::memcpy((void*) data1, image.data, sizeof(decltype(image.at<cv::Vec3b>(0, 0)[0])) * image.total());

        // end = CLOCK::now();
        // std::cout << "FPS #2 FORMING TENSOR AND NORMALIZATION " <<  (CLOCK_CAST(end - begin).count() / 1000.0) << std::endl;
        end = CLOCK::now();
        time_preproc = time_preproc + (CLOCK_CAST(end - begin).count() / 1000.0);
        t_preproc = (CLOCK_CAST(end - begin).count() / 1000.0);
        if (t_preproc > t_preproc_max) {
            t_preproc_max = t_preproc;
        }
        if (t_preproc < t_preproc_min) {
            t_preproc_min = t_preproc;
        }
        std::cout << "FPS #1 PREPROCESSING: " <<  (CLOCK_CAST(end - begin).count() / 1000.0) << std::endl;

        begin = CLOCK::now();
        infer_request.infer();                      // INFER
        end = CLOCK::now();
        time_infer = time_infer + (CLOCK_CAST(end - begin).count() / 1000.0);
        t_infer = (CLOCK_CAST(end - begin).count() / 1000.0);
        if (t_infer > t_infer_max) {
            t_infer_max = t_infer;
        }
        if (t_infer < t_infer_min) {
            t_infer_min = t_infer;
        }
        std::cout << "FPS #2 INFERENCE: " <<  (CLOCK_CAST(end - begin).count() / 1000.0) << std::endl;

        begin = CLOCK::now();                       // PARSING BOXES
        ov::Tensor output_tensor1 = infer_request.get_output_tensor(0);
        ov::Tensor output_tensor2 = infer_request.get_output_tensor(1);
        
        auto out_data1 = output_tensor1.data<float_t>();
        auto box_shape1 = output_tensor1.get_shape();
        auto box_type = output_tensor1.get_element_type();

        auto out_data2 = output_tensor2.data<float_t>();
        auto mask_shape2 = output_tensor2.get_shape();

        std::cout << "ouputs shapes: " << box_shape1 << " and " << mask_shape2 << std::endl;

        std::vector<Detection_mask> detections = parsing_boxes(image, out_data1, box_shape1, box_type, modelScoreThreshold, modelNMSThreshold, number_classes, name_classes);
        // end = CLOCK::now();
        // std::cout << "FPS #4 PARSING BBOXES " <<  (CLOCK_CAST(end - begin).count() / 1000.0) << std::endl;
        
        torch::Tensor proto;
        torch::Tensor mask_in;
        torch::Tensor rect;
        torch::Tensor mask_in_m;
        torch::Tensor matrix_multi;
        torch::Tensor matrix_multi_3d;

        cv::Mat zer_mask;
        std::vector<cv::Mat> zer_masks;
        std::vector<cv::Mat> mat_multi_3d;
        
        // auto segment_shape;
        auto image_shape = {640,640};
        int detections_num = detections.size();

        if (detections_num) {
            // begin = CLOCK::now();                   // FROM OUTOUT TO TORCH TENSOR
            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            proto = torch::from_blob(out_data2, {32,160,160}, options);
            mask_in = torch::zeros({detections_num, 32}, {torch::kFloat32});
            rect = torch::zeros({detections_num, 4}, {torch::kFloat32});
            // end = CLOCK::now();
            // std::cout << "FPS #4 DATA TO TTENSOR " <<  (CLOCK_CAST(end - begin).count() / 1000.0) << std::endl;
            
            // begin = CLOCK::now();                   // FROM [0,640] pixels to [0,160] pixels
            for(int num = 0; num < detections_num; num++) {
                for(int mask_elem = 0; mask_elem < 32; mask_elem++) {
                    mask_in[num][mask_elem] = (detections[num].mask[mask_elem]);
                    
                }
                detections[num].box.x = std::clamp(detections[num].box.x / 4, 0, 160);
                detections[num].box.y = std::clamp(detections[num].box.y / 4, 0, 160);
                
                detections[num].box.width = detections[num].box.width / 4;
                if ((detections[num].box.x + detections[num].box.width) > 160 || (detections[num].box.x + detections[num].box.width) < 0)
                    detections[num].box.width = 160 - detections[num].box.x;
                
                detections[num].box.height = detections[num].box.height / 4;
                if ((detections[num].box.y + detections[num].box.height) > 160 || (detections[num].box.y + detections[num].box.height) < 0)
                    detections[num].box.height = 160 - detections[num].box.y;
            }
            // end = CLOCK::now();
            // std::cout << "FPS #5 TRANSFORM BBOXES " <<  (CLOCK_CAST(end - begin).count() / 1000.0) << std::endl;
            
            // begin = CLOCK::now();                   // MATRIX OPERATIONS
            mask_in_m = proto.view({32, 25600});
            matrix_multi = torch::mm(mask_in, mask_in_m);
            matrix_multi_3d = torch::sigmoid(matrix_multi).view({detections_num, 160, 160});

            // end = CLOCK::now();
            // std::cout << "FPS #6 MM " <<  (CLOCK_CAST(end - begin).count() / 1000.0) << std::endl;

            // begin = CLOCK::now();                   // FROM TORCH TENSORS TO MASKS
            for(int num = 0; num < detections_num; num++) {
                zer_masks.push_back(cv::Mat::zeros(160, 160, CV_32FC1));
                mat_multi_3d.push_back(cv::Mat::zeros(160, 160, CV_32FC1));
                std::cout << "after pushback" << std::endl;
                std::memcpy((void*) mat_multi_3d[num].data, matrix_multi_3d[num].data_ptr(), sizeof(decltype(matrix_multi_3d[num][0][0]))*(matrix_multi_3d[num].numel())/2);
                std::cout << "after memcpy matrix_multi_3d[num] = " << sizeof(matrix_multi_3d[num][0][0]) << " mat_multi_3d[num] = " << sizeof(decltype(mat_multi_3d[num])) << std::endl;
                std::cout << "roi.x | " << detections[num].box.x << std::endl;
                std::cout << "roi.y | " << detections[num].box.x << std::endl;
                std::cout << "roi.w | " << detections[num].box.width << std::endl;
                std::cout << "roi.h | " << detections[num].box.height << std::endl;
                if (detections[num].box.x != 0 &&  detections[num].box.y != 0 && detections[num].box.width !=0 && detections[num].box.height != 0 )
                    mat_multi_3d[num](cv::Rect(detections[num].box.x, detections[num].box.y, detections[num].box.width, detections[num].box.height)).copyTo(zer_masks[num](cv::Rect(detections[num].box.x, detections[num].box.y, detections[num].box.width, detections[num].box.height)));
                std::cout << "after Rect" << std::endl;
                if (num > 0) {
                    cv::bitwise_or(zer_masks[num], zer_masks[num-1], zer_masks[num]);
                }
            }
            // end = CLOCK::now();
            // std::cout << "FPS #7 ZEROS TO MASK " <<  (CLOCK_CAST(end - begin).count() / 1000.0) << std::endl;

            // begin = CLOCK::now();                   // DRAWING BBOXES
            // for (int i = 0; i < detections_num; ++i)
            // {
            //     Detection_mask detection = detections[i];

            //     cv::Rect box = detection.box;
            //     cv::Scalar color = detection.color;

            //     // Detection box text
            //     std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
            //     cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
            //     cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

            //     cv::rectangle(image, textBox, color, cv::FILLED);
            //     cv::putText(image, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
            // }
            cv::resize(zer_masks[detections_num-1], zer_masks[detections_num-1], scale, cv::INTER_CUBIC);
            // end = CLOCK::now();
            // std::cout << "FPS #9 DRAWING " <<  (CLOCK_CAST(end - begin).count() / 1000.0) << std::endl;
            std::cout << "NUMBER OF DETECTIONS: " << detections_num << std::endl;
        }
        // begin = CLOCK::now();                       // SHOWING BBOXES AND SEGMANTATION MASK
        cv::resize(zer_masks[detections_num-1], zer_masks[detections_num-1], cv::Size(720,540), cv::INTER_CUBIC);
        for(int i=0; i < 160; i++) {
            for(int j=0; j < 160; j++) {
                if(zer_masks[detections_num-1].at<float>(i,j) < 0)
                    zer_masks[detections_num-1].at<float>(i,j) = 0.0;
                if(zer_masks[detections_num-1].at<float>(i,j) > 1)
                    zer_masks[detections_num-1].at<float>(i,j) = 1.0;
                zer_masks[detections_num-1].at<float>(i,j) = std::round(zer_masks[detections_num-1].at<float>(i,j));
            }
        }
        // cv::imshow("webcam", image);
        // cv::imshow("test_mask", test_mask);
        // if(detections_num) {
        //     cv::imshow("test_mask", zer_masks[detections_num-1]);
        //     zer_masks[detections_num-1].convertTo(test_mask, CV_8UC1, 255);
            // break;
        // }
        // if(cv::waitKey(27)>=0)

        //     break;
        // end = CLOCK::now();
        // std::cout << "FPS #10 SHOWING " <<  (CLOCK_CAST(end - begin).count() / 1000.0) << std::endl;

        // end_global = CLOCK::now();
        // std::cout << "FPS FPS FPS FPS HERE " << 1.0/(CLOCK_CAST(end_global - begin_global).count() / 1000000.0) << std::endl;
        end = CLOCK::now();
        time_postproc = time_postproc + (CLOCK_CAST(end - begin).count() / 1000.0);
        t_postproc = (CLOCK_CAST(end - begin).count() / 1000.0);
        if (t_postproc > t_postproc_max) {
            t_postproc_max = t_postproc;
        }
        if (t_postproc < t_postproc_min) {
            t_postproc_min = t_postproc;
        }
        std::cout << "FPS #3 POSTPROCESSING: " <<  (CLOCK_CAST(end - begin).count() / 1000.0) << std::endl;
        numcycles=numcycles+1;
        // cv::imwrite("/home/ss21mipt/DIPLOMA/IoU_tool/YOLOV8_segmentation.jpg", test_mask);
        std::cout <<"NUM OF CYCLES " << numcycles << std::endl;
    }
    end_for_average = CLOCK::now();
    std::cout << "AVERAGE FPS: " << numcycles*1.0/(CLOCK_CAST(end_for_average - begin_for_average).count() / 1000000.0) << std::endl;
    std::cout << "AVERAGE PREPROCESSING FPS: " << numcycles*1.0/(time_preproc / 1000.0) << std::endl;
    std::cout << "AVERAGE INFERENCE FPS: " << numcycles*1.0/(time_infer / 1000.0) << std::endl;
    std::cout << "AVERAGE POSTPROCESSING FPS: " << numcycles*1.0/(time_postproc / 1000.0) << std::endl;

    std::cout << "AVERAGE PREPROCESSING time: " << ((time_preproc) / (numcycles*1.0)) << " in range [" << t_preproc_min << ", " << t_preproc_max << "]" << std::endl;
    std::cout << "AVERAGE INFERENCE time: " << ((time_infer) / (numcycles*1.0)) << " in range [" << t_infer_min << ", " << t_infer_max << "]" << std::endl;
    std::cout << "AVERAGE POSTPROCESSING time: " << ((time_postproc) / (numcycles*1.0)) << " in range [" << t_postproc_min << ", " << t_postproc_max << "]" << std::endl;

    // cv::destroyAllWindows();

}