#include <Eigen/Core>
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

    cv::Size modelShape(640,640);
    bool yolov8;
    std::cout << "BOX SHAPE" << box_shape1 << std::endl;
    // std::cout << "SEGMENTATION MASK" << mask_shape2 << std::endl;
    // std::cout << "SEGMENTATION MASK" << *out_data2 << std::endl;
    
    auto rows = box_shape1[1];
    auto dimensions = box_shape1[2];
    std::cout << "rows and dimensions and type " << rows << " " << dimensions << " " << box_type << std::endl;

    cv::Size s(dimensions,rows);
    cv::Mat output_boxes(s, CV_32FC1);

    std::cout << "RESULTS are Here: " << " " <<output_boxes.channels() << output_boxes.size[0] << std::endl;

    auto maxim_data = 0;
    for(int row = 0; row < rows; row++) {
        for(int dimension = 0; dimension < dimensions; dimension++) {
    
            output_boxes.at<float>(row,dimension) = out_data1[dimensions*row + dimension];
    
        }
    }
    std::cout << "maxim_data: " << maxim_data << std::endl;
    std::cout << "RESULTS are Here: " << " " <<output_boxes.size[0] << " " << output_boxes.size[1] << " " << output_boxes.size[3] <<std::endl;
    

    if (dimensions > rows) // Check if the shape[2] is more than shape[1] (yolov8)
    {
        yolov8 = true;
        rows = box_shape1[2];
        dimensions = box_shape1[1];

        output_boxes = output_boxes.reshape(1, dimensions);
        cv::transpose(output_boxes, output_boxes);
    }
    std::cout << "SUMMARY are Here: " << " " <<output_boxes.size[0] << " " << output_boxes.size[1] << image.cols << image.rows << std::endl;
    
    float *data = (float *)output_boxes.data;

    float x_factor = image.cols / modelShape.width;
    float y_factor = image.rows / modelShape.height;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<float *> detection_masks;
    
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
                
                std::cout << "[x,y,w,h]: [" << x << ", " << y << ", " << w << ", " << h << "] confidences: " << *confidences.rbegin() << "others are :" << dimensions <<std::endl;

                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);

                int width = int(w * x_factor);
                int height = int(h * y_factor);

                boxes.push_back(cv::Rect(left, top, width, height));
                detection_masks.push_back(data + 4 + number_classes);
            }
        }

        data += dimensions;
    }

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
    std::cout << net->input(0).get_partial_shape() << output1_shape << std::endl;

    input.tensor().set_element_type(ov::element::f32);   // NAPISAL PODRYGOMY (NE NHWC) почему здесь другая нумерация, как это может быть правильно написанным вариантом?
    input.model().set_layout("NHWC");
    
    net = ppp.build();
    
    ov::CompiledModel compiled_model = core.compile_model(net, "CPU");    // auto executable_network = ie.LoadNetwork(net, device);
    ov::InferRequest infer_request = compiled_model.create_infer_request();    // infer_request = executable_network.CreateInferRequest();
 
    ov::Tensor m_inputData = infer_request.get_input_tensor(0);    // m_inputData = infer_request.GetBlob(inputsName);
    
    std::cout << "Shape of input tensor: " << m_inputData.get_shape() << std::endl; 
    std::cout << "Type of input tensor: " << m_inputData.get_element_type() << std::endl;    
    auto m_numChannels = m_inputData.get_shape()[1];
    auto m_inputW = m_inputData.get_shape()[3];
    auto m_inputH = m_inputData.get_shape()[2];
    auto m_imageSize = m_inputH * m_inputW;
    auto data1 = m_inputData.data<float_t>();
    std::cout << "m_numChannels: " << m_numChannels <<  std::endl <<"m_inputW: " << m_inputW <<  std::endl << "m_inputH: " << m_inputH <<  std::endl<< "m_imageSize: " << m_imageSize << std::endl;

    cv::VideoCapture cap(0);
    cv::Mat image;
    cv::Mat segments;
    cv::Mat mask;
    cv::Mat test_mask;
    std::cout << "doshlo" << std::endl;
    bool yolov8;
    cv::Size modelShape(640,640);
    auto modelScoreThreshold = 0.35;
    auto modelNMSThreshold = 0.5;
    int number_classes = 1;
    std::string name_classes = "g";

    while (cap.isOpened()){
        // cap >> image;
        image = cv::imread(png);
        std::cout << "doshlo" << std::endl;
        if (image.empty() || !image.data) {
            return false;
        }
        cv::Size scale(640, 640);  
        cv::resize(image, image, scale);    
        
        std::cout << "SIZES of Mat: "  << image.size[0] << " " << image.size[1] << " " << image.channels()<<  std::endl;
  
        // FILLING THE DATA1
        
        for (size_t row = 0; row < m_inputH; row++) {
            for (size_t col = 0; col < m_inputW; col++) {
                for (size_t ch = 0; ch < m_numChannels; ch++) {

                    data1[m_imageSize * ch + row * m_inputW + col] = float(image.at<cv::Vec3b>(row, col)[ch] / 255.0);

                }
            }
        }

        infer_request.infer();

        ov::Tensor output_tensor1 = infer_request.get_output_tensor(0);
        ov::Tensor output_tensor2 = infer_request.get_output_tensor(1);
        

        auto out_data1 = output_tensor1.data<float_t>();
        auto box_shape1 = output_tensor1.get_shape();
        auto box_type = output_tensor1.get_element_type();

        auto out_data2 = output_tensor2.data<float_t>();
        auto mask_shape2 = output_tensor2.get_shape();

        std::cout << "SEGMENTATION MASK" << mask_shape2 << std::endl;

        

        // PARSING BOXES

        std::vector<Detection_mask> detections = parsing_boxes(image, out_data1, box_shape1, box_type, modelScoreThreshold, modelNMSThreshold, number_classes, name_classes);

        // PARSING SEGMENTATION



        int detections_num = detections.size();
        std::cout << "Number of detections:" << detections_num << std::endl;

        for (int i = 0; i < detections_num; ++i)
        {
            Detection_mask detection = detections[i];

            cv::Rect box = detection.box;
            cv::Scalar color = detection.color;
            std::cout << "MASKS" << std::endl;  
            for (int mix = 0; mix < 32; mix++) {
                std::cout << "MASK[" << mix << "] = " << *(detection.mask + mix) << std::endl;  
                
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
        // Inference ends here...


        cv::imshow("webcam", image);
        if(cv::waitKey(30)>=0)
            break;
    }
    cv::destroyAllWindows();
}