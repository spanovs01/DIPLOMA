#include "Filters/Custom/YoloV8OpenVino.hpp"
#include "CameraState/CameraState.hpp"
#include "Filters/Patches/PatchProvider.hpp"
#include "Utils/RotatedRectUtils.hpp"
#include "Utils/Interface.h"
#include "Utils/OpencvUtils.h"
#include "Utils/ROITools.hpp"
#include "rhoban_utils/timing/benchmark.h"

#include "rhoban_geometry/circle.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <utility>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <rhoban_utils/logging/logger.h>
#include <torch/script.h>

#include <opencv2/dnn/dnn.hpp>

#include <hl_monitoring/field.h>

static rhoban_utils::Logger logger("YoloV8OpenVino");

using namespace std;
using namespace rhoban_geometry;
using ::rhoban_utils::Benchmark;


static inline  bool SortScorePairDescend(const std::pair<float, int>& pair1,
                          const std::pair<float, int>& pair2)
{
    return pair1.first > pair2.first;
}

static inline double jaccardDistance(const cv::Rect& a, const cv::Rect& b) {
    double Aa = a.area();
    double Ab = b.area();

    if ((Aa + Ab) <= std::numeric_limits<double>::epsilon()) {
        // jaccard_index = 1 -> distance = 0
        return 0.0;
    }

    double Aab = (a & b).area();
    // distance = 1 - jaccard_index
    return 1.0 - Aab / (Aa + Ab - Aab);
}

static inline float rectOverlap(const cv::Rect& a, const cv::Rect& b)
{
    return 1.f - static_cast<float>(jaccardDistance(a, b));
}

namespace Vision
{
namespace Filters
{
// std::map<std::string, hl_monitoring::Field::POIType> YoloV5OpenVino::stringToPOIEnum = {
//   { "ArenaCorner", hl_monitoring::Field::POIType::ArenaCorner },
//   { "LineCorner", hl_monitoring::Field::POIType::LineCorner },
//   { "T", hl_monitoring::Field::POIType::T },
//   { "X", hl_monitoring::Field::POIType::X },
//   { "Center", hl_monitoring::Field::POIType::Center },
//   { "PenaltyMark", hl_monitoring::Field::POIType::PenaltyMark },
//   { "PostBase", hl_monitoring::Field::POIType::PostBase }
// };

YoloV8OpenVino::YoloV8OpenVino() : Filter("YoloV8OpenVino"), model_path("yolov8_segm.xml")
{
  // TODO load classes from json config file

  // WARNING order is important (alphabetical order)
  //   if classes are removed, just comment the corresponding line, make sure to keep alphabetical order(!now Empty first)
  //classNames.push_back("ArenaCorner");
 // classNames.push_back("Empty");
//   classNames.push_back("Ball");
//   classNames.push_back("Center");
//   classNames.push_back("Empty");
//   classNames.push_back("LineCorner");
//   classNames.push_back("PenaltyMark");
//   classNames.push_back("PostBase");
//   classNames.push_back("Robot");
//   classNames.push_back("T");
//   classNames.push_back("X");
}

void YoloV8OpenVino::setParameters()
{
  debugLevel = ParamInt(0, 0, 1);
  scoreThreshold = ParamFloat(0.6, 0.0, 1.0);
  nmsThreshold = ParamFloat(0.6, 0.0, 1.0);
  confThreshold = ParamFloat(0.8, 0.0, 1.0);
  //imSize = ParamInt(32, 1, 64);

  params()->define<ParamInt>("debugLevel", &debugLevel);
  params()->define<ParamFloat>("scoreThreshold", &scoreThreshold);
  params()->define<ParamFloat>("nmsThreshold", &nmsThreshold);
  params()->define<ParamFloat>("confThreshold", &confThreshold);
  //params()->define<ParamInt>("imSize", &imSize);

//   for (const std::string& className : classNames)
//   {
//     isUsingFeature[className] = ParamInt(0, 0, 1);
//     params()->define<ParamInt>("uses" + className, &(isUsingFeature[className]));
//   }
}

// void YoloV5OpenVino::updateUsedClasses()
// {
//   usedClassNames.clear();
//   for (size_t idx = 0; idx < classNames.size(); idx++)
//   {
//     const std::string& className = classNames[idx];
//     if (className == "Empty" || isUsingFeature[className] != 0)
//     {
//       usedClassNames.push_back(className);
//     }
//   }
// }

std::string YoloV8OpenVino::getClassName() const
{
  return "YoloV8OpenVino";
}

Json::Value YoloV8OpenVino::toJson() const
{
  Json::Value v = Filter::toJson();
  v["model_path"] = model_path;
  v["labels_path"] = labels_path;
  v["device"] = device;
  return v;
}

void YoloV8OpenVino::fromJson(const Json::Value& v, const std::string& dir_name)
{
  Filter::fromJson(v, dir_name);
  rhoban_utils::tryRead(v, "model_path", &model_path);
  rhoban_utils::tryRead(v, "labels_path", &labels_path);
  rhoban_utils::tryRead(v, "device", &device);

  init();
}

int YoloV8OpenVino::expectedDependencies() const
{
  return 1;
}

void YoloV8OpenVino::init()
{
    net = ie.ReadNetwork(model_path);
    inputs = net.getInputsInfo();
    outputs = net.getOutputsInfo();
    for (auto item : inputs)
    {
        inputsName = item.first;
        auto input_data = item.second;
        input_data->setPrecision(InferenceEngine::Precision::FP32);
        input_data->setLayout(InferenceEngine::Layout::NCHW);
        input_data->getPreProcess().setColorFormat(InferenceEngine::ColorFormat::RGB);
        std::cout << "set precision for inputsName: " << inputsName << std::endl;
    }
 
 
    for (auto item : outputs)
    {
        auto output_data = item.second;
        std::string outputName = item.first;
        output_data->setPrecision(InferenceEngine::Precision::FP32);
        // outputsName = item.first;
        std::cout << "set precision for outputsName: " << outputName << std::endl;
        outputsName.push_back(std::move(outputName));

    }
    std::cout << "Loading model to the device " << device << std::endl;
    executableNet = ie.LoadNetwork(net, device);
    infer_request = executableNet.CreateInferRequest();
 
    m_inputData = infer_request.GetBlob(inputsName);
    m_1dim = m_inputData->getTensorDesc().getDims()[0];
    m_numChannels = m_inputData->getTensorDesc().getDims()[1];
    m_inputW = m_inputData->getTensorDesc().getDims()[2];
    m_inputH = m_inputData->getTensorDesc().getDims()[3];

    m_imageSize = m_inputH * m_inputW;

    net.setBatchSize(1);
    /** Reading labels (if specified) **/
    std::ifstream inputFile(labels_path);
    std::copy(std::istream_iterator<std::string>(inputFile),
                std::istream_iterator<std::string>(),
                std::back_inserter(labels));

    numClasses = labels.size();
}

double YoloV8OpenVino::sigmoid(double x){
    return (1 / (1 + exp(-x)));
}

std::vector<int> YoloV8OpenVino::getAnchors(int net_grid){
    std::vector<int> anchors(6);
    int a80[6] = {10,13, 16,30, 33,23};
    int a40[6] = {30,61, 62,45, 59,119};
    int a20[6] = {116,90, 156,198, 373,326}; 
    if(net_grid == 80){
        anchors.insert(anchors.begin(),a80,a80 + 6);
    }
    else if(net_grid == 40){
        anchors.insert(anchors.begin(),a40,a40 + 6);
    }
    else if(net_grid == 20){
        anchors.insert(anchors.begin(),a20,a20 + 6);
    }
    return anchors;
}

void YoloV8OpenVino::GetMaxScoreIndex(const std::vector<float>& scores, const float threshold, const int top_k,
                      std::vector<std::pair<float, int> >& score_index_vec)
{
    // Generate index score pairs.
    for (size_t i = 0; i < scores.size(); ++i)
    {
        if (scores[i] > threshold)
        {
            score_index_vec.push_back(std::make_pair(scores[i], i));
        }
    }

    // Sort the score pair according to the scores in descending order
    std::stable_sort(score_index_vec.begin(), score_index_vec.end(),
                     SortScorePairDescend);

    // Keep top_k scores if needed.
    if (top_k > 0 && top_k < (int)score_index_vec.size())
    {
        score_index_vec.resize(top_k);
    }
}

void YoloV8OpenVino::NMSBoxes(const std::vector<cv::Rect>& bboxes,
      const std::vector<float>& scores, const float score_threshold,
      const float nms_threshold,
      std::vector<int>& indices,
      int limit = std::numeric_limits<int>::max())
{
    float eta = 1.f;
    int top_k = 0;
    // Get top_k scores (with corresponding indices).
    std::vector<std::pair<float, int> > score_index_vec;
    GetMaxScoreIndex(scores, score_threshold, top_k, score_index_vec);

    // Do nms.
    float adaptive_threshold = nms_threshold;
    indices.clear();
    for (size_t i = 0; i < score_index_vec.size(); ++i) {
        const int idx = score_index_vec[i].second;
        bool keep = true;
        for (int k = 0; k < (int)indices.size() && keep; ++k) {
            const int kept_idx = indices[k];
            float overlap = rectOverlap(bboxes[idx], bboxes[kept_idx]);
            keep = overlap <= adaptive_threshold;
        }
        if (keep) {
            indices.push_back(idx);
            if (indices.size() >= limit) {
                break;
            }
        }
        if (keep && eta < 1 && adaptive_threshold > 0.5) {
          adaptive_threshold *= eta;
        }
    }
}

// void YoloV5OpenVino::segment(std::vector<Detection_mask> &outputs, cv::Mat &maska)
// {
    
// }
void YoloV8OpenVino::detect(cv::Mat &image, std::vector<Detection_mask> &outputs, cv::Mat &mask)
{
    // cv::Mat blob_image;
    cv::Mat blob_image = cv::imread("/home/rhoban/env/common/Feds_yolov8/img.jpg");
    std::cout << "blob_image is read" << std::endl;
    // int number_classes = numClasses;
    int number_classes = 1;
    std::cout << "numClasses = " << numClasses << std::endl;
    // cv::resize(image, blob_image, cv::Size(m_inputW, m_inputH));
    cvtColor(blob_image, blob_image, cv::COLOR_BGR2RGB);
 
    float* data = static_cast<float*>(m_inputData->buffer());
    for (size_t row = 0; row < m_inputH; row++) {
        for (size_t col = 0; col < m_inputW; col++) {
            for (size_t ch = 0; ch < m_numChannels; ch++) {
#ifdef NCS2
				data[m_imageSize * ch + row * m_inputW + col] = float(blob_image.at<cv::Vec3b>(row, col)[ch]);
#else
				data[m_imageSize * ch + row * m_inputW + col] = float(blob_image.at<cv::Vec3b>(row, col)[ch] / 255.0);
#endif // NCS2
            }
        }
    }
    
    infer_request.Infer();
    // auto output = infer_request.GetBlob(outputsName);
    output0 = infer_request.GetBlob(outputsName[0]);
    size_t fdim = output0->getTensorDesc().getDims()[0];
    size_t sdim = output0->getTensorDesc().getDims()[1];
    size_t tdim = output0->getTensorDesc().getDims()[2];
    std::cout << "output0 sizes: dim0 = " << fdim << " ;    dim1 = " << sdim << " ;    dim2 = " << tdim << " ;" << std::endl;
    output1 = infer_request.GetBlob(outputsName[1]);
    fdim = output1->getTensorDesc().getDims()[0];
    sdim = output1->getTensorDesc().getDims()[1];
    tdim = output1->getTensorDesc().getDims()[2];
    size_t fourdim = output1->getTensorDesc().getDims()[3];
    std::cout << "output1 sizes: dim0 = " << fdim << " ;    dim1 = " << sdim << " ;    dim2 = " << tdim << " ;  dim3 = " << fourdim << " ;" << std::endl;
	// const float* detection_out = static_cast<PrecisionTrait<Precision::FP32>::value_type*>(output->buffer());
	float* detection0_out = static_cast<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>(output0->buffer());
    float* detection1_out = static_cast<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>(output1->buffer());

    // const InferenceEngine::SizeVector outputDims = output->getTensorDesc().getDims();
    const InferenceEngine::SizeVector output0_Dims = output0->getTensorDesc().getDims();
    cv::Size modelShape(640,640);
    auto dimensions = output0_Dims[1];
    auto rows = output0_Dims[2];
    cv::Size s(dimensions,rows);
    cv::Mat output_boxes = cv::Mat(s, CV_32FC1, detection0_out);
    output_boxes = output_boxes.reshape(1, dimensions);
    cv::transpose(output_boxes, output_boxes);
    float *dataout = (float *)detection0_out;
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<float>> detection_masks;
    std::vector<float> mass;

    for (int i = 0; i < rows; ++i) {
        float *classes_scores = dataout+4;
        cv::Mat scores(number_classes, labels.size(), CV_32FC1, classes_scores);
        cv::Point class_id;
        double max_class_score;
        cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
        
        if (max_class_score > scoreThreshold)
        {
            confidences.push_back(max_class_score);
            class_ids.push_back(class_id.x);

            float x = dataout[0];
            float y = dataout[1];
            float w = dataout[2];
            float h = dataout[3];
            int left = int((x - 0.5 * w));
            int top = int((y - 0.5 * h));
            int width = int(w);
            int height = int(h);
            boxes.push_back(cv::Rect(left, top, width, height));
            for (int index=0; index < 32; index++) {
                mass.push_back(*(dataout + 4 + number_classes + index));
            }
            detection_masks.push_back(mass);
            
        }
        dataout += dimensions;
    }

    vector<int> nms_result;
    NMSBoxes(boxes, confidences, scoreThreshold, nmsThreshold, nms_result);
    for (int i = 0; i < nms_result.size(); i++)
    {
        int idx = nms_result[i];
        Detection_mask result;
        result.class_id = class_ids[idx];
        result.label = labels[result.class_id];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        result.mask = detection_masks[idx];
		outputs.push_back(result);
    }
    std::cout << "Starting of torch initialisations" << std::endl;
    torch::Tensor proto;
    torch::Tensor mask_in;
    torch::Tensor mask_in_m;
    torch::Tensor matrix_multi;
    torch::Tensor matrix_multi_3d;
    std::vector<cv::Mat> zer_masks;
    std::vector<cv::Mat> mat_multi_3d;
    cv::Mat zer_mask;
    cv::Mat maska = cv::Mat::zeros(160, 160, CV_32FC1);
    int detections_num = outputs.size();
    std::cout << "Starting of torch operations" << std::endl;
    if (detections_num > 0) {
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        proto = torch::from_blob(detection1_out, {32,160,160}, options);
        mask_in = torch::zeros({detections_num, 32}, {torch::kFloat32});
        std::cout << "Starting of (for) with [0,640] pixels to [0,160] pixels " << std::endl;
        for(int num = 0; num < detections_num; num++) {     // FROM [0,640] pixels to [0,160] pixels
            for(int mask_elem = 0; mask_elem < 32; mask_elem++) {
                mask_in[num][mask_elem] = (outputs[num].mask[mask_elem]);
                
            }
            outputs[num].box.x = std::clamp(outputs[num].box.x / 4, 0, 160);
            outputs[num].box.y = std::clamp(outputs[num].box.y / 4, 0, 160);
            
            outputs[num].box.width = outputs[num].box.width / 4;
            if ((outputs[num].box.x + outputs[num].box.width) > 160 || (outputs[num].box.x + outputs[num].box.width) < 0)
                outputs[num].box.width = 160 - outputs[num].box.x;
            
            outputs[num].box.height = outputs[num].box.height / 4;
            if ((outputs[num].box.y + outputs[num].box.height) > 160 || (outputs[num].box.y + outputs[num].box.height) < 0)
                outputs[num].box.height = 160 - outputs[num].box.y;
        }
        std::cout << "Starting of torch calculations" << std::endl;
        mask_in_m = proto.view({32, 25600});
        std::cout << "view is changed" << std::endl;
        matrix_multi = torch::mm(mask_in, mask_in_m);
        std::cout << "matrix multiplication is done" << std::endl;
        matrix_multi_3d = torch::sigmoid(matrix_multi).view({detections_num, 160, 160});
        std::cout << "End of torch calculations" << std::endl;
        
        for(int num = 0; num < detections_num; num++) {
            zer_masks.push_back(cv::Mat::zeros(160, 160, CV_32FC1));
            mat_multi_3d.push_back(cv::Mat::zeros(160, 160, CV_32FC1));
            std::cout << "after pushback" << std::endl;
            std::memcpy((void*) mat_multi_3d[num].data, matrix_multi_3d[num].data_ptr(), sizeof(decltype(matrix_multi_3d[num][0][0]))*(matrix_multi_3d[num].numel())/2);
            mat_multi_3d[num](cv::Rect(outputs[num].box.x, outputs[num].box.y, outputs[num].box.width, outputs[num].box.height)).copyTo(zer_masks[num](cv::Rect(outputs[num].box.x, outputs[num].box.y, outputs[num].box.width, outputs[num].box.height)));
            std::cout << "after Rect" << std::endl;
            if (num > 0) {
                cv::bitwise_or(zer_masks[num], zer_masks[num-1], zer_masks[num]);
            }
        }
        std::cout << "summary bitwising" << std::endl;
        cv::bitwise_or(maska, zer_masks[detections_num-1], maska);
    }

}

void YoloV8OpenVino::process()
{
    clearAllFeatures();
    frame = (*(getDependency().getImg())).clone();
    std::vector<Detection_mask> objects;
    cv::Mat maska = cv::Mat::zeros(160, 160, CV_32FC1);
    detect(frame, objects, maska);
    // segment(objects, mask);
    // for (auto &object : objects) {
    //     if (object.confidence >= scoreThreshold) {
    //         /** Drawing only objects when >confidence_threshold probability **/

    //         if (object.label == "ball") {
    //             cv::Point2f object_center(object.box.x + object.box.width/2, object.box.y + object.box.height/2);
    //             pushBall(object_center);
    //         }
    //         else if (object.label == "robot") {
    //             cv::Point2f object_botom_center(object.box.x + object.box.width/2, object.box.y + object.box.height);
    //             pushRobot(object_botom_center);
    //         }
    //         else if (object.label == "goalpost") {
    //             cv::Point2f object_botom_center(object.box.x + object.box.width/2, object.box.y + object.box.height);
    //             pushPOI(hl_monitoring::Field::POIType::PostBase, object_botom_center);
    //         }
    //         else
    //             logger.error("Unknown label: '%s'", object.label);
    //         cv::putText(frame,
    //                 (object.class_id < static_cast<int>(labels.size()) ?
    //                         object.label : std::string("label #") + std::to_string(object.class_id)) + std::string(" ") + std::to_string(object.confidence),
    //                     cv::Point2f(static_cast<float>(object.box.x), static_cast<float>(object.box.y - 5)), cv::FONT_HERSHEY_COMPLEX_SMALL, 1,
    //                     cv::Scalar(0, 0, 255));
    //         cv::rectangle(frame, object.box, cv::Scalar(0, 0, 255));
    //     }
    // }
    //std::cerr << "RETURNING IMAGE " << std::endl;
    std::cout << "AFTER DETECT()" << std::endl;
    cv::resize(maska, maska, cv::Size(720,540));
    std::cout << "AFTER RESIZE()" << std::endl;
    img() = maska;
    std::cout << "END OF FILTER" << std::endl;
    
    
    // if (isModeChanged) {
    //     isModeChanged = false;
    // }

    // // Final point:
    // // in the truly Async mode, we swap the NEXT and CURRENT requests for the next iteration
    // frame = next_frame;
    // next_frame = cv::Mat();
    // if (isAsyncMode) {
    //     async_infer_request_curr.swap(async_infer_request_next);
    // }

}
}  // namespace Filters
}  // namespace Vision
