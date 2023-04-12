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
    float * mask;
};



std::vector<Detection_mask> parsing_boxes(cv::Mat image, auto out_data1, auto box_shape1, auto box_type, float modelScoreThreshold, float modelNMSThreshold, int number_classes, std::string &name_classes) {

    std::time_t time0;
    std::time_t time1;
    std::time_t time2;
    std::time_t time3;
    // auto time4;

    cv::Size modelShape(640,640);
    bool yolov8;
    // std::cout << "BOX SHAPE" << box_shape1 << std::endl;
    // std::cout << "SEGMENTATION MASK" << mask_shape2 << std::endl;
    // std::cout << "SEGMENTATION MASK" << *out_data2 << std::endl;
    
    auto rows = box_shape1[1];
    auto dimensions = box_shape1[2];
    // std::cout << "rows and dimensions and type " << rows << " " << dimensions << " " << box_type << std::endl;

    cv::Size s(dimensions,rows);
    cv::Mat output_boxes(s, CV_32FC1);

    // std::cout << "RESULTS are Here: " << " " <<output_boxes.channels() << output_boxes.size[0] << std::endl;

    auto maxim_data = 0;
    time0 = std::time(nullptr);
    for(int row = 0; row < rows; row++) {
        for(int dimension = 0; dimension < dimensions; dimension++) {
    
            output_boxes.at<float>(row,dimension) = out_data1[dimensions*row + dimension];
    
        }
    }
    time1 = std::time(nullptr);

    std::cout << "Time for filling output_boxes: " << time1 - time0 << std::endl;
    // std::cout << "RESULTS are Here: " << " " <<output_boxes.size[0] << " " << output_boxes.size[1] << " " << output_boxes.size[3] <<std::endl;
    

    if (dimensions > rows) // Check if the shape[2] is more than shape[1] (yolov8)
    {
        yolov8 = true;
        rows = box_shape1[2];
        dimensions = box_shape1[1];

        output_boxes = output_boxes.reshape(1, dimensions);
        cv::transpose(output_boxes, output_boxes);
    }
    // std::cout << "SUMMARY are Here: " << " " <<output_boxes.size[0] << " " << output_boxes.size[1] << image.cols << image.rows << std::endl;
    
    float *data = (float *)output_boxes.data;

    float x_factor = image.cols / modelShape.width;
    float y_factor = image.rows / modelShape.height;
    // std::cout << "X AND Y FACTORS" << x_factor<< " " << y_factor << std::endl;
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<float *> detection_masks;
    
    time2 = std::time(nullptr);
    for (int i = 0; i < rows; ++i)
    {
        if (yolov8)
        {
            float *classes_scores = data+4;

            cv::Mat scores(1, number_classes, CV_32FC1, classes_scores);
            cv::Point class_id;
            double maxClassScore;

            minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

            if (maxClassScore > modelScoreThreshold)
            {
                confidences.push_back(maxClassScore);
                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                
                // std::cout << "[x,y,w,h]: [" << x << ", " << y << ", " << w << ", " << h << "] confidences: " << *confidences.rbegin() << "others are :" << dimensions <<std::endl;

                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);

                int width = int(w * x_factor);
                int height = int(h * y_factor);
                // std::cout << "[left,top,wisth,height]: [" << left << ", " << top << ", " << width << ", " << height << "] confidences: " << *confidences.rbegin() << "others are :" << dimensions <<std::endl;

                boxes.push_back(cv::Rect(left, top, width, height));
                detection_masks.push_back(data + 4 + number_classes);
            }
        }

        data += dimensions;
    }
    time3 = std::time(nullptr);

    std::cout << "Time for filling boxes and detectiond_masks: " << time3 - time2 << std::endl;

    std::vector<int> nms_result;
    // NMSBoxes(boxes, confidences, modelScoreThreshold, modelNMSThreshold, nms_result);
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
    std::time_t time4;
    std::time_t time5;
    // auto xml = "/home/ss21mipt/Documents/starkit/DIPLOMA/YOEO/config/IR&onnx_for_416_Petr_1/yoeo.xml";
    // auto xml = "/home/ss21mipt/Documents/starkit/DIPLOMA/to_rhoban/weights/Feds_yolov8_2_openvino/best.xml";
    auto xml = "/home/ss21mipt/DIPLOMA/weights/best_openvino_model/best.xml";
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
    
    // std::cout << "Shape of input tensor: " << m_inputData.get_shape() << std::endl; 
    // std::cout << "Type of input tensor: " << m_inputData.get_element_type() << std::endl;    
    auto m_numChannels = m_inputData.get_shape()[1];
    auto m_inputW = m_inputData.get_shape()[3];
    auto m_inputH = m_inputData.get_shape()[2];
    auto m_imageSize = m_inputH * m_inputW;
    auto data1 = m_inputData.data<float_t>();
    // std::cout << "m_numChannels: " << m_numChannels <<  std::endl <<"m_inputW: " << m_inputW <<  std::endl << "m_inputH: " << m_inputH <<  std::endl<< "m_imageSize: " << m_imageSize << std::endl;

    cv::VideoCapture cap(0);
    cv::Mat image;
    cv::Mat segments;
    cv::Mat mask;
    cv::Mat test_mask;
    // std::cout << "doshlo" << std::endl;
    bool yolov8;
    cv::Size modelShape(640,640);
    auto modelScoreThreshold = 0.35;
    auto modelNMSThreshold = 0.5;
    int number_classes = 1;
    std::string name_classes = "g";

    while (cap.isOpened()){
        cap >> image;
        // image = cv::imread(png);
        // std::cout << "doshlo" << std::endl;
        if (image.empty() || !image.data) {
            return false;
        }
        cv::Size scale(640, 640);  
        cv::resize(image, image, scale);    
        
        // std::cout << "SIZES of Mat: "  << image.size[0] << " " << image.size[1] << " " << image.channels()<<  std::endl;
  
        // FILLING THE DATA1
        time4 = std::time(nullptr);
        for (size_t row = 0; row < m_inputH; row++) {
            for (size_t col = 0; col < m_inputW; col++) {
                for (size_t ch = 0; ch < m_numChannels; ch++) {

                    data1[m_imageSize * ch + row * m_inputW + col] = float(image.at<cv::Vec3b>(row, col)[ch] / 255.0);

                }
            }
        }
        time5 = std::time(nullptr);

        std::cout << "filling data1: " << time5 -time4 << std::endl;

        infer_request.infer();

        ov::Tensor output_tensor1 = infer_request.get_output_tensor(0);
        ov::Tensor output_tensor2 = infer_request.get_output_tensor(1);
        

        auto out_data1 = output_tensor1.data<float_t>();
        auto box_shape1 = output_tensor1.get_shape();
        auto box_type = output_tensor1.get_element_type();

        auto out_data2 = output_tensor2.data<float_t>();
        auto mask_shape2 = output_tensor2.get_shape();

        // std::cout << "SEGMENTATION MASK" << mask_shape2 << std::endl;

        

        // PARSING BOXES

        std::vector<Detection_mask> detections = parsing_boxes(image, out_data1, box_shape1, box_type, modelScoreThreshold, modelNMSThreshold, number_classes, name_classes);


        // PARSING SEGMENTATION


        
        // out_data2

        torch::Tensor proto;
        torch::Tensor mask_in;
        torch::Tensor rect;
        torch::Tensor mask_in_m;
        torch::Tensor matrix_multi;
        torch::Tensor matrix_multi_3d;
        torch::Tensor r;
        torch::Tensor c;
        std::vector<cv::Mat> test_masks;
        cv::Mat many_masks;
        cv::Mat test_mask_pred;
        // auto segment_shape;
        auto image_shape = {640,640};
        int detections_num = detections.size();
        // std::cout << "Number of detections:" << detections_num << std::endl;
        if (detections_num) {
            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            // torch::Tensor proto = torch::zeros({32,160,160});
            proto = torch::from_blob(out_data2, {32,160,160}, options);
            // std::cout << "PROTO FILLED" << std::endl;
            mask_in = torch::zeros({detections_num, 32}, {torch::kFloat32});
            rect = torch::zeros({detections_num, 4}, {torch::kFloat32});
            for(int num = 0; num < detections_num; num++) {
                for(int mask_elem = 0; mask_elem < 32; mask_elem++) {
                    mask_in[num][mask_elem] = *(detections[num].mask + mask_elem);
                    
                }
                rect[num][0] = detections[num].box.x * 0.25;
                rect[num][1] = detections[num].box.y * 0.25;
                rect[num][2] = (detections[num].box.x + detections[num].box.width) * 0.25;
                rect[num][3] = (detections[num].box.y + detections[num].box.height) * 0.25;
                // std::cout << "Rect elements!!! = " << rect[num][0] << " " << rect[num][1] << " " << rect[num][2] << " " << rect[num][3] << " " <<std::endl;
                
            }
            // std::cout << "MASK_IN AND RECT FILLED" << std::endl;
            
            mask_in_m = proto.view({32, 25600});

            matrix_multi = torch::mm(mask_in, mask_in_m);
            matrix_multi_3d = torch::sigmoid(matrix_multi).view({detections_num, 160, 160});
            rect = rect.unsqueeze(2);
            auto bbs = torch::chunk(rect, 4, 1);
            r = torch::arange(160).unsqueeze(0).unsqueeze(0);
            c = torch::arange(160).unsqueeze(0).unsqueeze(2);
            // std::cout << "R and C shapes: " << r.sizes() << " " << c.sizes() << std::endl;
            auto maxim = 0;
            auto minim = 250;
            for(int num = 0; num < detections_num; num++) {
                for (int h = 0; h < 160; h++) {
                    for (int w = 0; w < 160; w++) {
                        if (((r[0][0][w].item<float>() >= bbs[0][num][0].item<float>()) && (r[0][0][w].item<float>() < bbs[2][num][0].item<float>()) && (c[0][h][0].item<float>() >= bbs[1][num][0].item<float>()) && (c[0][h][0].item<float>() < bbs[3][num][0].item<float>())) == false) {
                            matrix_multi_3d[num][h][w] = 0;    
                        }
                    }
                }
            }

            
            // std::cout << "matrix_multi_3d " << matrix_multi_3d.sizes() << " Rect " << rect.sizes() << std::endl;
            
            for(int num = 0; num < detections_num; num++) {
                test_masks.push_back(cv::Mat::zeros(160, 160, CV_32FC1));
                for(size_t i=0; i<160; i++){
                    for(size_t j=0; j<160; j++){
                        test_masks[num].at<float>(j,i) = (float)(matrix_multi_3d[num][j][i].item<float>());
                    } 
                }
                if (num > 0) {
                    cv::bitwise_or(test_masks[num], test_masks[num-1], test_masks[num]);
                }
            }

            for (int i = 0; i < detections_num; ++i)
            {
                Detection_mask detection = detections[i];

                cv::Rect box = detection.box;
                cv::Scalar color = detection.color;
                // std::cout << "MASKS" << std::endl;  
                for (int mix = 0; mix < 32; mix++) {
                    // std::cout << "MASK[" << mix << "] = " << *(detection.mask + mix) << std::endl;  
                    
                }  
                // Detection box
                cv::rectangle(image, box, color, 2);

                // Detection box text
                std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
                cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
                cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

                cv::rectangle(image, textBox, color, cv::FILLED);
                cv::putText(image, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
            }
            cv::resize(test_masks[detections_num-1], test_masks[detections_num-1], scale);
        }
        cv::imshow("webcam", image);
        // cv::imshow("test_mask", test_mask);
        if(detections_num) {
            cv::imshow("test_mask", test_masks[detections_num-1]);

        }
        // cv::imshow("test_mask1", test_masks[1]);
        if(cv::waitKey(30)>=0)
            break;
    }
    cv::destroyAllWindows();
}