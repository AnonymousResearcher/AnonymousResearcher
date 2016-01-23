#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include <string>
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include <iostream>
int rnncount = 0;
int displaycount = 1;
float local_decay = 1.0 * 0.0005;
int cccccc = 0;
int snapshot_ = 2000;

int jixu = 0;


#if 0
layers {
  bottom: "bigfea"
  bottom: "label_language_wang"
  bottom: "label_weak"
  top: "rnn"
  top: "label_rnn"
  top: "rnn_neg"
  name: "rnn"
  type: WANGRNN
  wangrnn_param {
    max_relation：30
    history_: true
    max_label: 10
    rnn_weight_mean: 0
    rnn_weight_var: 0.05
  }
}
#endif



namespace caffe {



  


template <typename Dtype>
void WangRnnLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const int max_relation = this->layer_param_.wangrnn_param().max_relation();
  const int max_labels = this->layer_param_.wangrnn_param().max_labels();
  const int total_relation = this->layer_param_.wangrnn_param().total_relation();
  addr = this->layer_param_.wangrnn_param().addr();
  num_ = bottom[0]->num(); 
  channels_ = bottom[0]->channels();//21
  height_ = bottom[0]->height();//1
  width_ = bottom[0]->width();//4096

  const int top_num = num_;
  const int top_channels = width_;
  const int top_height = 1; 
  const int top_width = max_relation;

  top[0]->Reshape(top_num, top_channels, top_height, top_width);
  top[1]->Reshape(top_num, 1, top_height, top_width);


  bottom_diff_wang.Reshape(num_, channels_, height_, width_);

  const int fealen = bottom[0]->width();

  rnn_weight.Reshape(1, 1, 2 * fealen, fealen);
  rnn_weight_history.Reshape(1, 1, 2 * fealen, fealen);
  rnn_weight2.Reshape(1, 1, 1, fealen);
  rnn_weight2_history.Reshape(1, 1, 1, fealen);
  
  const Dtype rnn_weight_mean = this->layer_param_.wangrnn_param().rnn_weight_mean();
  const Dtype rnn_weight_var = this->layer_param_.wangrnn_param().rnn_weight_var();

  const bool history_ = this->layer_param_.wangrnn_param().history_();
  
  if(!history_){
    caffe_rng_gaussian<Dtype>(fealen * 2 * fealen, rnn_weight_mean, rnn_weight_var, 
    rnn_weight.mutable_cpu_data());

    caffe_rng_gaussian<Dtype>(fealen, rnn_weight_mean, rnn_weight_var, 
      rnn_weight2.mutable_cpu_data());
  }else{
    LOG(INFO)<< "fituning form the history !";
    const int test_iter = this->layer_param_.wangrnn_param().test_iter();
    std::string filename = addr + "/rnn_weight";
    const int kBufferSize = 20;
    char iter_str_buffer[kBufferSize];
    snprintf(iter_str_buffer, kBufferSize, "_iter_%d", test_iter);
    filename += iter_str_buffer;
    std::ifstream weight_if(filename.c_str());

    
    std::string line;
    Dtype* w0 = rnn_weight.mutable_cpu_data();
    Dtype* w2 = rnn_weight2.mutable_cpu_data();
    for (int hh = 0; hh < 2 * fealen; hh++) {
      for(int ww = 0; ww < fealen; ww++){
        weight_if >> line;
        float val=atof(line.c_str());
        w0[hh * 2 * fealen + ww] = val;       
      }
  }

  std::string filenameone = addr + "/rnn_weight2";
  const int kBufferSizeone = 20;
  char iter_str_bufferone[kBufferSizeone];
  snprintf(iter_str_bufferone, kBufferSizeone, "_iter_%d", test_iter);
  filenameone += iter_str_bufferone;
  std::ifstream weight2_if(filenameone.c_str());
  std::string line2;
  for(int ww = 0; ww < fealen; ww++){
    weight2_if >> line2;
    float val = atof(line2.c_str());
    w2[ww] = val;
  }
  weight_if.close();
  weight2_if.close();
  }
  
  //LOG(INFO)<<"WWWWWWWWWWWWWWWWW";

  tmp_bottom.Reshape(num_,max_relation,1,2 * fealen);
  tmp_top.Reshape(num_,max_relation,1,fealen);

  error_bottom.Reshape(num_,max_relation,1,2 * fealen);
  error_top.Reshape(num_,max_relation,1,fealen);

  error_bottom_tmp.Reshape(1,1,1,2 * fealen);
  error_top_tmp.Reshape(1,1,1,fealen);
}

template <typename Dtype>
void WangRnnLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void WangRnnLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

 
  
  const int max_relation = this->layer_param_.wangrnn_param().max_relation();
  const int top_num = num_;
  const int top_channels = width_;
  const int top_height = 1; 
  const int top_width = max_relation;
  const int fealen = top_channels; 

  memset(top[0]->mutable_cpu_data(), 0, sizeof(Dtype) * top[0]->count());
  memset(top[1]->mutable_cpu_data(), 0, sizeof(Dtype) * top[1]->count());
  memset(tmp_bottom.mutable_cpu_data(), 0, sizeof(Dtype) * tmp_bottom.count());
  memset(tmp_top.mutable_cpu_data(), 0, sizeof(Dtype) * tmp_top.count());
  memset(error_bottom.mutable_cpu_data(), 0, sizeof(Dtype) * error_bottom.count());
  memset(error_top.mutable_cpu_data(), 0, sizeof(Dtype) * error_top.count());
  memset(tmp_top.mutable_cpu_diff(), 0, sizeof(Dtype) * tmp_top.count());
  memset(error_top.mutable_cpu_diff(), 0, sizeof(Dtype) * error_top.count());
  Dtype score_sum = 0;
  Dtype scorexian = 0;
  Dtype scorexian2 = 0;
  bigerror_lib.clear();

  for (int n = 0; n < num_; ++n) {
    bool doublejump = false;
    Dtype* bottom_dataone = bottom[1]->mutable_cpu_data(n,0,0,0);//语言标签
    vector<vector<int> > error_lib;
    error_lib.clear();
    
    vector<vector<int> > treestruct;


    // bool others = false;
    for(int tree_ii = 0; tree_ii < max_relation; tree_ii++){

      if(bottom_dataone[tree_ii * 3] > -1){

          vector<int> aa;
          aa.clear();
          aa.push_back(bottom_dataone[tree_ii * 3]);
          aa.push_back(bottom_dataone[tree_ii * 3 + 1]);
          aa.push_back(bottom_dataone[tree_ii * 3 + 2 ]);
          // if(bottom_dataone[tree_ii * 3 + 2 ] == 200){
          //   others = true;
          // }
          treestruct.push_back(aa);
      }
    }
    

    // if (!treestruct.size()) {
    //   bigerror_lib.push_back(error_lib);
    //   LOG(INFO) << "this is a empty relationship  !!!!!!";
    //   continue;
    // }

    // if (treestruct.size() == 1 &&treestruct[0][2] == 201 ) {
    //   bigerror_lib.push_back(error_lib);
    //   LOG(INFO) << "this is a background category !!!!!!";
    //   continue;
    // }

    // if (others) {
    //   bigerror_lib.push_back(error_lib);
    //   LOG(INFO) << "this is a others category !!!!!!";
    //   continue;
    // }

    vector<int> keywords;
    keywords.clear();
    
    for (int j = 0; j < 10; ++j) {
      const int label = static_cast<int>(*bottom[2]->cpu_data(n, j));
      if (label == -1) {
          continue;
      } else if (label >= 0 && label < 21) {
        keywords.push_back(label);
      } else {
        LOG(FATAL) << "Unexpected label " << label;
      }
    }
    // Do nothing if the image has no label
    if (keywords.size() == 0) {
      bigerror_lib.push_back(error_lib);
      LOG(INFO) << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++";
      continue;
    }
    vector<int> tmp_labels(keywords);

    // for(int ii =0;ii < keywords.size();ii++){
    //   Dtype* ceshi = bottom[0]->mutable_cpu_data(n,keywords[ii],0,0);
    //   for(int jj =0;jj < 10;jj++){
    //     LOG(INFO)<< ceshi[jj];
    //   }
    //   LOG(INFO)<<"------------------------";
    // }

      //按语言树来聚合，求每次聚合的得分，以及记录错误聚合时，每次聚合得分最高的那个

      for(int tree_ii = 0; tree_ii < treestruct.size (); tree_ii++){

        int c0 = treestruct[tree_ii][0];
        int c1 = treestruct[tree_ii][1];
        int c2 = treestruct[tree_ii][2];//映射成top[0]的第几类
        
        vector<int>::iterator jiance =std::find(keywords.begin(),keywords.end(),
        c0);

        vector<int>::iterator jiance2 =std::find(keywords.begin(),keywords.end(),
        c1);

        //如果实体被裁掉了，那这个三元组就忽略了
        if((jiance==keywords.end()&&c0<200)||(jiance2==keywords.end()&& c1<200)){
          LOG(INFO)<<"CUT to be lost";
          doublejump = true;
          continue;
        }
        int flag_c0 = -1;
        if (c0 > 199){
          flag_c0 = findrelation(c0, treestruct, tree_ii);
        }

        int flag_c1 = -1; 
        if (c1 > 199){
          flag_c1 = findrelation(c1, treestruct,tree_ii);
        }

        Dtype score;
        if (flag_c0 == -1){
          LOG(INFO)<< "single layers!";
          Dtype* tmp_bottom_data = tmp_bottom.mutable_cpu_data(n,tree_ii,0,0);
          caffe_copy(fealen, bottom[0]->cpu_data(n,c0,0,0), tmp_bottom_data);
        }else if(flag_c0 == 10000){
          LOG(INFO) << "FATAL error !";
          continue;
        }else{
          LOG(INFO)<<"very good !";
          Dtype* tmp_bottom_data = tmp_bottom.mutable_cpu_data(n,tree_ii,0,0);
          caffe_copy(fealen, tmp_top.cpu_data(n,flag_c0,0,0),tmp_bottom_data);
        }
   
        if (flag_c1 == -1 ){
          Dtype* tmp_bottom_data = tmp_bottom.mutable_cpu_data(n,tree_ii,0,fealen);
          caffe_copy(fealen, bottom[0]->cpu_data(n,c1,0,0), tmp_bottom_data);
        }else if (flag_c1 == 10000){
          LOG(INFO)<< "FATAL error!";
          for(int ii = 0 ; ii < treestruct.size();ii++){
            LOG(INFO)<< treestruct[ii][0] <<" "<<treestruct[ii][1]<<" "<<treestruct[ii][2];
          }
          continue;
        }else{
          LOG(INFO)<<"very good ! ";
          Dtype* tmp_bottom_data = tmp_bottom.mutable_cpu_data(n,tree_ii,0,fealen);
          caffe_copy(fealen, tmp_top.cpu_data(n,flag_c1,0,0), tmp_bottom_data);        
        }         
          
        Dtype* tmptmp = new Dtype [fealen];
        Dtype* jihuo_dao = new Dtype [fealen];
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 1, fealen, fealen * 2 ,
          (Dtype) (1.), tmp_bottom.cpu_data(n,tree_ii,0,0), rnn_weight.cpu_data(), 
          (Dtype) 0., tmptmp);

        Dtype* tmp_top_data = tmp_top.mutable_cpu_data(n,tree_ii,0,0);
        for(int ii = 0; ii < fealen; ii++){
          Dtype vari = tmptmp[ii];
          tmp_top_data[ii] = 1/(1 + exp(-vari)) ;
          //tmp_top_data[ii] = (exp(vari)- exp(-vari))/(exp(vari)+ exp(-vari));
          jihuo_dao[ii] = tmp_top_data[ii] * (1-tmp_top_data[ii]);
          //jihuo_dao[ii] = 1 - tmp_top_data[ii] * tmp_top_data[ii];
        } 
        delete[] tmptmp;
        Dtype* top_data = top[0]->mutable_cpu_data(n,0,0,tree_ii);
        Dtype* top_data_1 = top[1]->mutable_cpu_data(n,0,0,tree_ii);
        

        for (int cc = 0; cc < fealen; cc ++){
          top_data[cc * top_height * top_width] = tmp_top_data[cc];
        }
        top_data_1[0] = c2 - 200;

        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 1, 1 , fealen,
          (Dtype) (1.), rnn_weight2.cpu_data(), tmp_top.cpu_data(n,tree_ii,0,0), 
          (Dtype) 0., &score);
        
         score_sum = score_sum - score;
         scorexian = score;

         //LOG(INFO)<< "------------------------score is : " << score;
        Dtype* top_diff_addr = tmp_top.mutable_cpu_diff(n,tree_ii,0,0);
        caffe_mul(fealen, rnn_weight2.cpu_data(), jihuo_dao, top_diff_addr);
        caffe_scal(fealen, Dtype(-1.), top_diff_addr);

        delete[] jihuo_dao;
        //caffe_copy(tmp_top.count(), rnn_weight2.cpu_data(), top_diff_addr);
        #if 1

        memset(error_bottom_tmp.mutable_cpu_data(), 0, sizeof(Dtype) * error_bottom_tmp.count());
        memset(error_top_tmp.mutable_cpu_data(), 0, sizeof(Dtype) * error_top_tmp.count());
        vector<std::pair<Dtype, vector<int> > > score_vector;
        score_vector.clear();
        for(int ll =0; ll < tmp_labels.size();ll++){
          for(int mm =0; mm < tmp_labels.size(); mm++){
            if(ll == mm){
              continue;
            }
            int c00 = tmp_labels[ll];
            int c11 = tmp_labels[mm];

            int flag_c00 = -1;
            if (c00 > 199){
              flag_c00 = findrelation(c00, treestruct,tree_ii);
            }
            int flag_c11 = -1;
            if(c11 > 199){
              flag_c11 = findrelation(c11, treestruct,tree_ii);
            }
            if (flag_c00 == -1){
              Dtype* error_bottom_data = error_bottom_tmp.mutable_cpu_data();
              caffe_copy(fealen, bottom[0]->cpu_data(n,c00,0,0), error_bottom_data);
            }else if(flag_c00 == 10000){
              LOG(INFO)<< "FATAL ERROR!";
              continue;
            }else{
              Dtype* error_bottom_data = error_bottom_tmp.mutable_cpu_data();
              caffe_copy(fealen, tmp_top.cpu_data(n,flag_c00,0,0), error_bottom_data);
            }
            if (flag_c11 == -1){
              Dtype* error_bottom_data = error_bottom_tmp.mutable_cpu_data(0,0,0,fealen);
              caffe_copy(fealen, bottom[0]->cpu_data(n,c11,0,0), error_bottom_data);
            }else if(flag_c11 == 10000){
              LOG(INFO)<< "FATAL ERROR!";
            }else{
              Dtype* error_bottom_data = error_bottom_tmp.mutable_cpu_data(0,0,0,fealen);
              caffe_copy(fealen, tmp_top.cpu_data(n,flag_c11,0,0), error_bottom_data);
            }
            Dtype* tmp = new Dtype [fealen];
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 1, fealen, fealen * 2 ,
              (Dtype) (1.), error_bottom_tmp.cpu_data(), rnn_weight.cpu_data(), 
              (Dtype) 0., tmp);

            Dtype* error_top_data = error_top_tmp.mutable_cpu_data();
            for(int ii = 0; ii < fealen; ii++){
              Dtype vari = tmp[ii];
              error_top_data[ii] = 1/(1 + exp(-vari)) ;
              //error_top_data[ii] = (exp(vari)- exp(-vari))/(exp(vari)+ exp(-vari));
              //error_jihuo_dao[ii] = error_top_data[ii] * (1-error_top_data[ii]);
              //error_jihuo_dao[ii] = 1 - error_top_data[ii] * error_top_data[ii];
            } 
            delete[] tmp;

            Dtype score_tmp;
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 1, 1 , fealen,
              (Dtype) (1.), rnn_weight2.cpu_data(), error_top_tmp.cpu_data(), 
              (Dtype) 0., &score_tmp);
            vector<int> aa;
            aa.clear();
            aa.push_back(c00);
            aa.push_back(c11);
            score_vector.push_back(std::make_pair(score_tmp, aa));
          }
        }
        for(vector<int>::iterator it = tmp_labels.begin();it!=tmp_labels.end();){
          if(*it== c0 || *it==c1)
            it=tmp_labels.erase(it);
          else
            it++;
        }
        tmp_labels.push_back(c2);
        Dtype score_max;
        std::partial_sort(score_vector.begin(), score_vector.begin() + 2,
          score_vector.end(), std::greater<std::pair<Dtype, std::vector<int> > >());
        score_sum = score_sum + score_max;
        scorexian2 = score_max;
        //LOG(INFO)<< "--------------------score mas is: " << score_max;

        if (!(score_vector[0].second[0] == c0 && score_vector[0].second[1] ==c1)){
          score_max = score_vector[0].first;
          vector<int> tmp_error;
          tmp_error.clear();
          tmp_error.push_back(score_vector[0].second[0]);
          tmp_error.push_back(score_vector[0].second[1]);
          tmp_error.push_back(c2);
          error_lib.push_back(tmp_error);

          int c000 = score_vector[0].second[0];
          int c111 = score_vector[0].second[1];

          int flag_c000 = -1;
          if (c000 > 199){
            flag_c000 = findrelation(c000, treestruct, tree_ii);
          }

          int flag_c111 = -1;
          if (c111 > 199){
            flag_c111 = findrelation(c111, treestruct, tree_ii);
          }


          if (flag_c000 == -1){
            Dtype* error_bottom_data = error_bottom.mutable_cpu_data(n,tree_ii,0,0);
            caffe_copy(fealen, bottom[0]->cpu_data(n,c000,0,0), error_bottom_data);
          }else if(flag_c000 == 10000){
            LOG(INFO)<<"FATAL ERROR !";
            continue;
          }else{
            Dtype* error_bottom_data = error_bottom.mutable_cpu_data(n,tree_ii,0,0);
            caffe_copy(fealen, tmp_top.cpu_data(n,flag_c000,0,0), error_bottom_data);
          }

          if (flag_c111 == -1){
            Dtype* error_bottom_data = error_bottom.mutable_cpu_data(n,tree_ii,0,fealen);
            caffe_copy(fealen, bottom[0]->cpu_data(n,c111,0,0), error_bottom_data);
          }else if(flag_c111 == 10000){
            LOG(INFO)<<"FATAL ERROR !";
            continue;
          }else{
            Dtype* error_bottom_data = error_bottom.mutable_cpu_data(n,tree_ii,0,fealen);
            caffe_copy(fealen, tmp_top.cpu_data(n,flag_c111,0,0), error_bottom_data);
          }

          Dtype* tmptmptmp = new Dtype [fealen];
          Dtype* error_jihuo_dao = new Dtype [fealen];
          caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 1, fealen, fealen * 2 ,
            (Dtype) (1.), error_bottom.cpu_data(n,tree_ii,0,0), rnn_weight.cpu_data(), 
            (Dtype) 0., tmptmptmp);

          Dtype* error_top_data = error_top.mutable_cpu_data(n,tree_ii,0,0);
          for(int ii = 0; ii < fealen; ii++){
            Dtype vari = tmptmptmp[ii];
            error_top_data[ii] = 1/(1 + exp(-vari)) ;
            //error_top_data[ii] = (exp(vari)- exp(-vari))/(exp(vari)+ exp(-vari));
            error_jihuo_dao[ii] = error_top_data[ii] * (1-error_top_data[ii]);
            //error_jihuo_dao[ii] = 1 - error_top_data[ii] * error_top_data[ii];
          } 
          delete[] tmptmptmp;
          Dtype* top_diff_addr_error = error_top.mutable_cpu_diff(n,tree_ii,0,0);
          caffe_mul(fealen, rnn_weight2.cpu_data(), error_jihuo_dao, top_diff_addr_error);
          delete[] error_jihuo_dao;
        }else{
          //do nothing
          LOG(INFO)<< "succed !!! ";
        }
        #endif
      
      }//tree的右括号
      bigerror_lib.push_back(error_lib);  
      if(doublejump){
        continue;
      }
  }//num的右括号
  
  if (rnncount % displaycount == 0){
    LOG(INFO)<< " ----------------- total score is : " << score_sum / num_;
    //LOG(INFO) << "score is : " << scorexian << "  " << scorexian2; 
  }
  
}//forward的右括号

//"感觉对于top,继续沿用bottom的width_等变量，十分挫"
//"RNN 的反向传播"
template <typename Dtype>
void WangRnnLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom){
  const int max_relation = this->layer_param_.wangrnn_param().max_relation();
  const int top_num = num_;
  const int top_channels = width_;
  const int top_height = 1; 
  const int top_width = max_relation;
  const int fealen = top_channels; 
  memset(tmp_bottom.mutable_cpu_diff(), 0, sizeof(Dtype) * tmp_bottom.count());
  memset(error_bottom.mutable_cpu_diff(), 0, sizeof(Dtype) * error_bottom.count());
  memset(bottom_diff_wang.mutable_cpu_data(), 0, sizeof(Dtype) * bottom_diff_wang.count());
  memset(rnn_weight.mutable_cpu_diff(), 0, sizeof(Dtype) * rnn_weight.count());
  memset(rnn_weight2.mutable_cpu_diff(), 0, sizeof(Dtype) * rnn_weight2.count());
  

  for(int n = 0; n < num_; n++){

    Dtype* bottom_dataone = bottom[1]->mutable_cpu_data(n,0,0,0);//语言标签
    // vector<int> keywords;
    // keywords.clear();
    // for (int j = 0; j < 10; ++j) {
    //   const int label = static_cast<int>(*bottom[2]->cpu_data(n, j));
    //   if (label == -1) {
    //       continue;
    //   } else if (label >= 0 && label < 21) {
    //     keywords.push_back(label);
    //   } else {
    //     LOG(FATAL) << "Unexpected label " << label;
    //   }
    // }
    // // Do nothing if the image has no label
    // if (keywords.size() == 0) { 
    //   continue;
    // }

    //读取语言标签
    vector<vector<int> > treestruct;
    // LOG(INFO) << "NUM IS : " << num_ << " bigerror_lib size is " << bigerror_lib.size() <<
    // " key size : " << keywords.size();
    vector<vector<int> > & error_lib = bigerror_lib[n];

    if(error_lib.size() ==0){
      continue;
    }
    for(int tree_ii = 0; tree_ii < max_relation; tree_ii++){

      if(bottom_dataone[tree_ii * 3] > -1){

          vector<int> aa;
          aa.clear();
          aa.push_back(bottom_dataone[tree_ii * 3]);
          aa.push_back(bottom_dataone[tree_ii * 3 + 1]);
          aa.push_back(bottom_dataone[tree_ii * 3 + 2 ]);
          treestruct.push_back(aa);
      }
    }

    //求临时残差
    for(int tree_ii = treestruct.size() -1; tree_ii > -1; tree_ii--){
      // Dtype lambda;
      // if (treestruct[tree_ii][2] == 200){
      //   lambda = 0.21;
      // }else if (treestruct[tree_ii][2] == 201){
      //   lambda = 0.1;

      // }else{
      //   lambda = 2.1;
      // }
      const Dtype* top_diff_addr = top[0]->cpu_diff(n,0,0,tree_ii);
      Dtype* get_top_diff = new Dtype [top_channels];
      for(int cc = 0; cc < top_channels; cc++){
        get_top_diff[cc] =  top_diff_addr[cc * top_height * top_width];
      }
      caffe_cpu_axpby(fealen, Dtype(1.), get_top_diff, Dtype(1), 
        tmp_top.mutable_cpu_diff(n,tree_ii,0,0));
      delete[] get_top_diff; 

      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 2 * fealen, 1, fealen,
          (Dtype) (1.), rnn_weight.cpu_data(), tmp_top.cpu_diff(n,tree_ii,0,0), 
          (Dtype) 0., tmp_bottom.mutable_cpu_diff(n,tree_ii,0,0));

      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 2 * fealen, 1, fealen,
          (Dtype) (1.), rnn_weight.cpu_data(), error_top.cpu_diff(n,tree_ii,0,0), 
          (Dtype) 0., error_bottom.mutable_cpu_diff(n,tree_ii,0,0));
      int c0 = treestruct[tree_ii][0];
      int c1 = treestruct[tree_ii][1];
      int flag_c0 = 0;
      if(c0 > 199){
        flag_c0 = findrelation(c0, treestruct, tree_ii);
        if(flag_c0 == 10000){
          LOG(INFO)<< "FATAL ERROR!";
        }else{
          caffe_cpu_axpby(fealen, Dtype(1.), tmp_bottom.cpu_diff(n,tree_ii,0,0), Dtype(1.), 
          tmp_top.mutable_cpu_diff(n,flag_c0,0,0));
        }
        
      }
      int flag_c1 = 0;
      if(c1 > 199){
        flag_c1 = findrelation(c1, treestruct, tree_ii);
        if(flag_c1 == 10000){
          LOG(INFO)<< "FATAL ERROR !";
        }else{
          caffe_cpu_axpby(fealen, Dtype(1.), tmp_bottom.cpu_diff(n,tree_ii,0,fealen), Dtype(1.), 
          tmp_top.mutable_cpu_diff(n,flag_c1,0,0));          
        }
      }

    }
    #if 0
    for(int tree_ii = 0; tree_ii < treestruct.size (); tree_ii++){
      //Dtype* bottom_diff_addr = tmp_bottom.mutable_cpu_diff(n,tree_ii,0,0);
      //int c2 = treestruct[tree_ii][2];
      const Dtype* top_diff_addr = top[0]->cpu_diff(n,0,0,tree_ii);
      Dtype* get_top_diff = new Dtype [top_channels];

      for(int cc = 0; cc < top_channels; cc++){
        get_top_diff[cc] = top_diff_addr[cc * top_height * top_width];
      }
      caffe_cpu_axpby(fealen, Dtype(1.), get_top_diff, Dtype(1.), 
        tmp_top.mutable_cpu_diff(n,tree_ii,0,0));
      delete[] get_top_diff; 
      //score
      #if 1
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 2 * fealen, 1, fealen,
          (Dtype) (1.), rnn_weight.cpu_data(), tmp_top.cpu_diff(n,tree_ii,0,0), 
          (Dtype) 0., tmp_bottom.mutable_cpu_diff(n,tree_ii,0,0));
      #endif
      #if 1
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 2 * fealen, 1, fealen,
          (Dtype) (1.), rnn_weight.cpu_data(), error_top.cpu_diff(n,tree_ii,0,0), 
          (Dtype) 0., error_bottom.mutable_cpu_diff(n,tree_ii,0,0));
      #endif

      }
      #endif
    //score
    #if 1
    for(int tree_ii = 0; tree_ii < treestruct.size(); tree_ii++){
      int c0 = treestruct[tree_ii][0];
      int c1 = treestruct[tree_ii][1];
      
      if(c0 < 200){
        Dtype* bottom_diff_ = bottom_diff_wang.mutable_cpu_data(n,c0,0,0);
        caffe_copy(fealen, tmp_bottom.mutable_cpu_diff(n,tree_ii,0,0), bottom_diff_);
      }
      if(c1 < 200){
      Dtype* bottom_diff_ = bottom_diff_wang.mutable_cpu_data(n,c1,0,0);
      caffe_copy(fealen, tmp_bottom.mutable_cpu_diff(n,tree_ii,0,fealen), bottom_diff_);
     }    
    }
    #endif
    #if 1
    for(int error_ii = 0; error_ii < error_lib.size(); error_ii++){
      int e0 = error_lib[error_ii][0];
      int e1 = error_lib[error_ii][1];
      
      if(e0 < 200){
        Dtype* bottom_diff_ = bottom_diff_wang.mutable_cpu_data(n,e0,0,0);
        caffe_copy(fealen, error_bottom.mutable_cpu_diff(n,error_ii,0,0), bottom_diff_);
      }
      if(e1 < 200){
        Dtype* bottom_diff_ = bottom_diff_wang.mutable_cpu_data(n,e1,0,0);
        caffe_copy(fealen, error_bottom.mutable_cpu_diff(n,error_ii,0,fealen), bottom_diff_);
      }
      
    }
    #endif
    
    //compute the difference of the weights
    for(int tree_ii = 0; tree_ii < treestruct.size();  tree_ii++){
      //score
      #if 1
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 2 * fealen, fealen, 1,
          (Dtype) (1.), tmp_bottom.cpu_data(n,tree_ii,0,0), tmp_top.cpu_diff(n,tree_ii,0,0), 
          (Dtype) 1., rnn_weight.mutable_cpu_diff());
      #endif
      #if 1
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 2 * fealen, fealen, 1,
          (Dtype) (1.), error_bottom.cpu_data(n,tree_ii,0,0), error_top.cpu_diff(n,tree_ii,0,0), 
          (Dtype) 1., rnn_weight.mutable_cpu_diff());
      #endif
      caffe_cpu_axpby(2*fealen * fealen, Dtype(local_decay), rnn_weight.mutable_cpu_data(),
        Dtype(1.), rnn_weight.mutable_cpu_diff());
      //score
      #if 1
      Dtype final_ = -1.;
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, fealen, 1, 1,
          (Dtype) (1.), tmp_top.cpu_data(n,tree_ii,0,0), &final_, 
          (Dtype) 1., rnn_weight2.mutable_cpu_diff());
      #endif
      #if 1
      Dtype final_error = 1.;
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, fealen, 1, 1,
          (Dtype) (1.), error_top.cpu_data(n,tree_ii,0,0), &final_error, 
          (Dtype) 1., rnn_weight2.mutable_cpu_diff());
      #endif

      caffe_cpu_axpby(fealen, Dtype(local_decay), rnn_weight2.mutable_cpu_data(),
        Dtype(1.), rnn_weight2.mutable_cpu_diff());

    }
  
  }
  caffe_copy(bottom[0]->count(), bottom_diff_wang.mutable_cpu_data(), 
    bottom[0]->mutable_cpu_diff());
  WangUpdate(); 
  if (rnncount % snapshot_ ==0 && rnncount!=0){
    WangSnapshot(fealen);
  }
  //Dtype* xianshi = bottom_diff_wang.mutable_cpu_data();
  rnncount ++ ;
}

template <typename Dtype>
void WangRnnLayer<Dtype>::WangUpdate(){
  Dtype local_rate = 0.001;
  Dtype momentum = 0.;

  caffe_cpu_axpby(rnn_weight.count(), local_rate,
                rnn_weight.cpu_diff(), momentum,
                rnn_weight_history.mutable_cpu_data());


  caffe_copy(rnn_weight.count(),
          rnn_weight_history.cpu_data(),
          rnn_weight.mutable_cpu_diff());

  caffe_cpu_axpby(rnn_weight2.count(), local_rate,
                rnn_weight2.cpu_diff(), momentum,
                rnn_weight2_history.mutable_cpu_data());


  caffe_copy(rnn_weight2.count(),
          rnn_weight2_history.cpu_data(),
          rnn_weight2.mutable_cpu_diff());


  caffe_cpu_axpby(rnn_weight.count(), Dtype(-1.), rnn_weight.cpu_diff(),
    Dtype(1.), rnn_weight.mutable_cpu_data());

  caffe_cpu_axpby(rnn_weight2.count(), Dtype(-1.), rnn_weight2.cpu_diff(),
    Dtype(1.), rnn_weight2.mutable_cpu_data());

  
}

template <typename Dtype>
void WangRnnLayer<Dtype>::WangSnapshot(const int fealen){

  std::string filename = addr + "/rnn_weight";
  const int kBufferSize = 20;
  char iter_str_buffer[kBufferSize];
  snprintf(iter_str_buffer, kBufferSize, "_iter_%d", rnncount);
  filename += iter_str_buffer;

  std::string filenameone = addr + "/rnn_weight2";
  const int kBufferSizeone = 20;
  char iter_str_bufferone[kBufferSizeone];
  snprintf(iter_str_bufferone, kBufferSizeone, "_iter_%d", rnncount);
  filenameone += iter_str_bufferone;

  printf("\nbegin to snapshot_:%s,%s",filename.c_str(), filenameone.c_str());

  std::ofstream wFile(filename.c_str());
  std::ofstream w2File(filenameone.c_str());

  Dtype* w0 = rnn_weight.mutable_cpu_data();
  Dtype* w2 = rnn_weight2.mutable_cpu_data();

  for (int hh = 0; hh < 2 * fealen; hh++) {
    for(int ww = 0; ww < fealen; ww++){
      wFile << w0[hh * 2 * fealen + ww] << " ";
    }
    wFile << "\n";
  }

  for(int ww = 0; ww < fealen; ww++){
    w2File << w2[ww] << " ";
  }
  wFile.close();
  w2File.close();
}

template <typename Dtype>
int WangRnnLayer<Dtype>::findrelation(int c0, vector<vector<int> > treestruct, int tree_ii){
  bool found = false;
  int lastfound = 0;
  for(int ii = 0; ii < tree_ii; ii++){
    if(c0 == treestruct[ii][2]){
      found = true;
      lastfound = ii;
    }
  }
  if(found == false){
    lastfound = 10000;
  }
  return lastfound;
}

INSTANTIATE_CLASS(WangRnnLayer);
REGISTER_LAYER_CLASS(WANGRNN, WangRnnLayer);

}  // namespace caffe
