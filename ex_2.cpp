#include<stdlib.h>
#include <iostream>   
#include <fstream>
#include <opencv2/core/core.hpp>   
#include <opencv2/highgui/highgui.hpp>  
#include<opencv2/imgproc/imgproc.hpp>
#include <opencv2\imgproc\imgproc_c.h>
#include <opencv2/features2d.hpp>
#include <vector>   
#include<algorithm>
#include <cmath> 
#include <assert.h>
#include<Windows.h>
using namespace std;
using namespace cv;


Mat Moravec(const char File_Name[], vector<Point>& Corner, int  threshold, int window);
void Feature_matching(const char file_name_L[], const char file_name_R[]);
//void Moravec(Mat M, vector<Point> /*&*/Corner, int  threshold, int window);
vector<Point> Corner, Corner_L , Corner_R ;

int main()
{
    char file_name_L[] = "D:\\大三上\\摄测\\课间实习二\\编程练习\\u0369_panLeft.bmp";
    char file_name_R[] = "D:\\大三上\\摄测\\课间实习二\\编程练习\\u0367_panRight.bmp";
    Feature_matching(file_name_L, file_name_R);

    return 0;
}

void writeToExcel(const char* filename, int threshold, int window, int cornerSize) {
    std::ofstream file(filename, std::ios_base::app);  // Open the file in append mode

    if (!file.is_open()) {
        std::cerr << "Error opening the file for writing." << std::endl;
        return;
    }

    // Write the values to the CSV file
    file << threshold << "," << window << "," << cornerSize << "\n";

    file.close();  // Close the file
}

Mat Moravec(const char File_Name[],vector<Point> &Corner,int  threshold,int window)
{
   
    //定义阈值大小
    clock_t start3, finish3;//定义时间函数
    double duration3;
    /* 测量一个事件持续的时间*/
    start3 = clock();//测试开始
    Mat img = imread(File_Name);
    //读取文件
    if (img.empty())
    {
        cout << File_Name << "文件打开失败\n";

    }
    else { cout << File_Name << "文件打开成功\n"; }
    // 判断文件是否正常打开  
    Mat M;
    if (img.type() == CV_8UC3) cvtColor(img, M, COLOR_BGR2GRAY, 0); else M = img;
    //如果是彩色图像，先转换为灰度图像处理
    int size = 7;
    int rows = M.rows, cols = M.cols, count_nums = 0, half = size / 2;
     vector<int> Value;
    //定义两个vector向量，用来储存角点Corner和对应的特征值Value
    for (int i = half; i < rows - half; i++)
    {
        for (int j = half; j < cols - half; j++)//按行和列进行像素遍历
        {

            //定义兴趣值容器，记录像素点四个方向的兴趣值
            int v1 = 0, v2 = 0, v3 = 0, v4 = 0, value = 0;

            for (int k = -half; k < half; k++)// 计算水平(0°)方向差值平方和
            {
                v1 += pow((M.at<uchar>(i, j + k) - M.at<uchar>(i, j + k + 1)), 2);
            }

            for (int k = -half; k < half; k++)// 计算45°方向差值平方和
            {
                v2 += pow((M.at<uchar>(i + k, j + k) - M.at<uchar>(i + k + 1, j + k + 1)), 2);
            }


            for (int k = -half; k < half; k++)// 计算垂直(90°)方向差值平方和
            {
                v3 += pow((M.at<uchar>(i + k, j) - M.at<uchar>(i + k + 1, j)), 2);
            }


            for (int k = -half; k < half; k++)// 计算135°方向差值平方和
            {
                v4 += pow((M.at<uchar>(i + k, j - k) - M.at<uchar>(i + k + 1, j - k - 1)), 2);
            }

            value = min(min(v1, v2), min(v3, v4));
            //取最小的兴趣值，作为像素点的特征值
            if (value > threshold)//进行阈值判断，大于制定阈值即确定为角点坐标
            {
                Corner.push_back(Point(j, i));//把角点坐标输入容器
                Value.push_back(value);//把特征值输入容器
            }

        }

    }
    //非极大值抑制法（筛选上面求出的角点，避免大量点聚集）

    for (int i = 0; i < Corner.size(); i++)
    {
        for (int j = i + 1; j < Corner.size(); )
        {
            if (abs(Corner[i].x - Corner[j].x) <= window / 2 && abs(Corner[i].y - Corner[j].y) <= window / 2)
            {//窗口内只取最明显的角点
                if (Value[i] >= Value[j])
                {
                    Value.erase(Value.begin() + j); Corner.erase(Corner.begin() + j);
                    //不明显的点消去
                }
                else {
                    Value.erase(Value.begin() + i); Corner.erase(Corner.begin() + i);
                    j = i + 1;

                }

            }
            else j++;

        }

    }

    finish3 = clock();//测试结束
    duration3 = (double)(finish3 - start3) / CLOCKS_PER_SEC;
    printf("此次事件耗时");
    printf("%f seconds\n", duration3);
    cout << "Moravec算法特征点个数 ：" << Corner.size() << endl << endl;
    writeToExcel("output.csv", threshold, window, Corner.size());
    for (int i = 0; i < Corner.size(); i++)
    {
        circle(img, Corner[i], 5, Scalar(0, 255, 255), 2, CV_AA);
        //特征点标号
        putText(img, to_string(i+1), Corner[i], FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(255, 0, 0), 1.8, CV_AA);

    }
    return img;
}
//void Moravec(Mat M, vector<Point> /*&*/Corner, int  threshold, int window)
//{
//
//    //定义阈值大小
//    clock_t start3, finish3;//定义时间函数
//    double duration3;
//    /* 测量一个事件持续的时间*/
//    start3 = clock();//测试开始
//    //Mat img = imread(File_Name);
//    ////读取文件
//    //if (img.empty())
//    //{
//    //    cout << File_Name << "文件打开失败\n";
//
//    //}
//    //else { cout << File_Name << "文件打开成功\n"; }
//    //// 判断文件是否正常打开  
//    //Mat M;
//    //if (img.type() == CV_8UC3) cvtColor(img, M, COLOR_BGR2GRAY, 0); else M = img;
//    //如果是彩色图像，先转换为灰度图像处理
//    int size = 7;
//    int rows = M.rows, cols = M.cols, count_nums = 0, half = size / 2;
//    vector<int> Value;
//    //定义两个vector向量，用来储存角点Corner和对应的特征值Value
//    for (int i = half; i < rows - half; i++)
//    {
//        for (int j = half; j < cols - half; j++)//按行和列进行像素遍历
//        {
//
//            //定义兴趣值容器，记录像素点四个方向的兴趣值
//            int v1 = 0, v2 = 0, v3 = 0, v4 = 0, value = 0;
//
//            for (int k = -half; k < half; k++)// 计算水平(0°)方向差值平方和
//            {
//                v1 += pow((M.at<uchar>(i, j + k) - M.at<uchar>(i, j + k + 1)), 2);
//            }
//
//            for (int k = -half; k < half; k++)// 计算45°方向差值平方和
//            {
//                v2 += pow((M.at<uchar>(i + k, j + k) - M.at<uchar>(i + k + 1, j + k + 1)), 2);
//            }
//
//
//            for (int k = -half; k < half; k++)// 计算垂直(90°)方向差值平方和
//            {
//                v3 += pow((M.at<uchar>(i + k, j) - M.at<uchar>(i + k + 1, j)), 2);
//            }
//
//
//            for (int k = -half; k < half; k++)// 计算135°方向差值平方和
//            {
//                v4 += pow((M.at<uchar>(i + k, j - k) - M.at<uchar>(i + k + 1, j - k - 1)), 2);
//            }
//
//            value = min(min(v1, v2), min(v3, v4));
//            //取最小的兴趣值，作为像素点的特征值
//            if (value > threshold)//进行阈值判断，大于制定阈值即确定为角点坐标
//            {
//                Corner.push_back(Point(j, i));//把角点坐标输入容器
//                Value.push_back(value);//把特征值输入容器
//            }
//
//        }
//
//    }
//    //非极大值抑制法（筛选上面求出的角点，避免大量点聚集）
//
//    for (int i = 0; i < Corner.size(); i++)
//    {
//        for (int j = i + 1; j < Corner.size(); )
//        {
//            if (abs(Corner[i].x - Corner[j].x) <= window / 2 && abs(Corner[i].y - Corner[j].y) <= window / 2)
//            {//窗口内只取最明显的角点
//                if (Value[i] >= Value[j])
//                {
//                    Value.erase(Value.begin() + j); Corner.erase(Corner.begin() + j);
//                    //不明显的点消去
//                }
//                else {
//                    Value.erase(Value.begin() + i); Corner.erase(Corner.begin() + i);
//                    j = i + 1;
//
//                }
//
//            }
//            else j++;
//
//        }
//
//    }
//
//    finish3 = clock();//测试结束
//    duration3 = (double)(finish3 - start3) / CLOCKS_PER_SEC;
//    printf("此次事件耗时");
//    printf("%f seconds\n", duration3);
//    cout << "Moravec算法特征点个数 ：" << Corner.size()<<" "<< threshold <<" "<< window << endl;
//    writeToExcel("output.csv", threshold, window, Corner.size());
//}


void Feature_matching(const char file_name_L[], const char file_name_R[])
{
    int window_size =7; int half = window_size / 2;//设置搜索窗口大小为7*7
    Mat img_L = imread(file_name_L); Mat M_L; if (img_L.type() == CV_8UC3) cvtColor(img_L, M_L, COLOR_BGR2GRAY, 0); else M_L = img_L;
    //for (int i = 1000; i < 5000; i += 100)
    //{
    //    for (int j = 5; j <55; j += 5)
    //    {
    //        Moravec(M_L, Corner_L,i,j);
    //    }
    //}
    Mat picture_L = Moravec(file_name_L, Corner_L,1000,40);
    namedWindow("Moravec", 0); imshow("Moravec", picture_L);
    waitKey(0); imwrite("Left.png", picture_L);
    //读入左影像，并调用Moravec函数求出特征点并记录
    Mat img_R = imread(file_name_R); Mat M_R; if (img_R.type() == CV_8UC3) cvtColor(img_R, M_R, COLOR_BGR2GRAY, 0); else M_R = img_R;
    Mat picture_R = Moravec(file_name_R, Corner_R, 1000, 40); namedWindow("Moravec", 0); imshow("Moravec", picture_R);
    waitKey(0); imwrite("Right.bmp", picture_R); // 读入右影像，并调用Moravec函数求出特征点并记录
    vector<Point> bestR_match, bestL_match;
    for (int k = 0; k < Corner_L.size(); k++)//k是特征点序号，依次遍历寻找同名点
    {
        double c = Corner_L[k].x, r = Corner_L[k].y;
        double lc = -175, lr = -6;//设置估计偏差
        double new_c = c + lc, new_r = r + lr;
        int big_window_size = 21;//设置滑动区域大小为21*21
        double max = 0;
        int bestR_i=0, bestR_j = 0, bestL_i = 0, bestL_j = 0;
        for (int i = -big_window_size / 2; i <= big_window_size / 2; i++)//在滑动区域内依次遍历，寻找合适的中心像素，i、j分别是滑动窗口的行列号
        {
            for (int j = -big_window_size / 2; j <= big_window_size / 2; j++)
            {
                double part_1 = 0, part_2 = 0, part_3 = 0, part_4 = 0, part_5 = 0;
                if (new_r + i - half > 0 && new_r + i+half < M_R.rows && new_c + j - half>0 && new_c + j+half < M_R.cols&& r -half > 0 && r + half < M_R.rows && c -half>0 && c + half < M_R.cols)
                {//计算7*7的搜索窗口中各数学变量的值
                    for (int m = -half; m <= half; m++)
                    {
                        for (int n = -half; n <= half; n++)
                        {
                            if (new_r + i - half > 0 && new_r + i+half < M_R.rows && new_c + j - half>0 && new_c + j+half < M_R.cols&& r -half > 0 && r + half < M_L.rows && c -half>0 && c + half < M_L.cols)
                            {//L_r和L_c是左图像的行列号，R_r和R_c是右图像的行列号
                                int L_r = r + m, L_c = c + n,R_r = new_r + i + m, R_c = new_c + j + n;
                                double L = M_L.at<uchar>(L_r, L_c), R = M_R.at<uchar>(R_r, R_c);
                                part_1 += L * R;
                                part_2 += L;
                                part_3 += R;
                                part_4 += L*L;
                                part_5 += R*R;
                                //由于相关系数公式计算较为复杂，拆分成5个部分进行计算避免混淆
                            }
                        }
                    }
                    double up = part_1 - ((part_2 * part_3) / pow(window_size, 2));
                    double down = ((part_4 - (part_2 * part_2/ pow(window_size, 2)))) * ((part_5 - (part_3 * part_3 / pow(window_size, 2))));
                    double correlation_coefficient = up / sqrt(down);//计算得到滑动窗口内的每个像素与匹配点的相关系数值
                    if (correlation_coefficient > max)//取最大的相关系数值进行记录
                    {
                        max = correlation_coefficient;
                        bestR_i = new_r + i;
                        bestR_j = new_c + j;
                        bestL_i = r;
                        bestL_j = c;
                    }
                  /*  cout << k+1<<" 行数："<< new_r + i <<" 列数"<< new_c + j<<" "<<correlation_coefficient << endl;*/
                }

            }


        }
        //cout << "-----------------------------------------------------------------------------------" << endl;
        //cout << "第" << k + 1 << "个点此时对应的行列（左）分别为" << r << " " << c << endl;
        //cout << "第" << k+1 << "个点相关系数最大为：" << max << "此时对应的行列（右）分别为" << best_i << " " << best_j<<endl;
        if (max > 0.85)//设置相关系数值阈值为0.85，大于0.85则判断为同名点
        {
            Point tempR, tempL; tempR.x = bestR_j; tempR.y = bestR_i, tempL.x = bestL_j; tempL.y = bestL_i;
            bestR_match.push_back(tempR); bestL_match.push_back(tempL);
        }
    }
    for (int i = 0; i < bestR_match.size(); i++)
    {
        circle(img_R, bestR_match[i], 5, Scalar(0, 255, 255), 1, CV_AA);
        //特征点标号
       /* putText(img_R, to_string(i + 1), Point(bestR_match[i].x-8, bestR_match[i].y+18), FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(255, 0, 0), 1.8, CV_AA);*/

        circle(img_L, bestL_match[i], 5, Scalar(0, 255, 255), 1, CV_AA);
        //特征点标号
      /* putText(img_L, to_string(i + 1), Point(bestL_match[i].x-8, bestL_match[i].y+18), FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(255, 0, 0), 1.8, CV_AA);*/

    }
    namedWindow("Moravec", 0); imshow("Moravec", img_R);
    waitKey(0); imwrite("match_R.bmp", img_R);
    namedWindow("Moravec", 0); imshow("Moravec", img_L);
    waitKey(0); imwrite("match_L.bmp", img_L);


    Mat combinedImage(max(img_L.rows, img_R.rows), img_L.cols + img_R.cols, img_L.type());//进行可视化，同名点连线
    img_L.copyTo(combinedImage(Rect(0, 0, img_L.cols, img_L.rows)));
    img_R.copyTo(combinedImage(Rect(img_L.cols, 0, img_R.cols, img_R.rows)));

    // 定义8种连线颜色
    vector<Scalar> colors;
    colors.push_back(Scalar(0, 0, 255));   // Red
    colors.push_back(Scalar(0, 255, 0));   // Green
    colors.push_back(Scalar(255, 0, 0));   // Blue
    colors.push_back(Scalar(0, 255, 255)); // Yellow
    colors.push_back(Scalar(255, 0, 255)); // Magenta
    colors.push_back(Scalar(255, 255, 0)); // Cyan
    colors.push_back(Scalar(255, 165, 0)); // Orange
    colors.push_back(Scalar(128, 0, 128)); // Purple


    for (size_t i = 0; i < bestL_match.size(); i++) {
        Point ptL = bestL_match[i];
        Point ptR = bestR_match[i];
        ptR.x += img_L.cols; 
        circle(combinedImage, ptL, 5, Scalar(0, 255, 0), 1, CV_AA); 
        Scalar color = colors[i % colors.size()]; 
        line(combinedImage, ptL, ptR, color, 1); 
    }

    namedWindow("Combined Image", 0);
    imshow("Combined Image", combinedImage);
    imwrite("Combined_Image.png", combinedImage);
    waitKey(0);


}











