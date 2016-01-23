#ifndef CAFFE_COMMON_LAYERS_HPP_
#define CAFFE_COMMON_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {


template <typename Dtype>
class WangL2Layer : public Layer<Dtype> {
 public:
  
  explicit WangL2Layer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_WANGL2;
  }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented (non-differentiable function)
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  
    // Do nothing
    // NOT_IMPLEMENTED;
  
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> s_;
  Blob<Dtype> diff_;

  int num_, channels_, height_, width_, channels_sec;
  bool out_max_val_;
  size_t top_k_;
  Blob<Dtype> bottom_diff_wang;
  vector<vector<std::pair<int,int> > > bigregion_size;
};

template <typename Dtype>
class WangAccLayer : public Layer<Dtype> {
 public:

  explicit WangAccLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_WANGACC;

  }
  virtual inline int ExactNumBottomBlobs() const { return 4; }
  virtual inline int ExactNumTopBlobs() const { return 0; }


 protected:
  
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented (non-differentiable function)
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    // Do nothing
    // NOT_IMPLEMENTED;
  }
  int num_, channels_, height_, width_;
  bool out_max_val_;
  size_t top_k_;
  int max_labels;
  Blob<Dtype> predict_;

};




template <typename Dtype>
class WangRegionLayer : public Layer<Dtype> {
 public:
  
  explicit WangRegionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_WANGREGION;

  }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 2; }


 protected:
  
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented (non-differentiable function)
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    // Do nothing
    // NOT_IMPLEMENTED;
  }
  int num_, channels_, height_, width_;
  bool out_max_val_;
  size_t top_k_;
  int max_labels;

};


template <typename Dtype>
class WangRegiontestLayer : public Layer<Dtype> {
 public:
  
  explicit WangRegiontestLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_WANGREGIONTEST;

  }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 2; }


 protected:
  
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented (non-differentiable function)
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    // Do nothing
    // NOT_IMPLEMENTED;
  }
  int num_, channels_, height_, width_;
  bool out_max_val_;
  size_t top_k_;
  int max_labels;

};



template <typename Dtype>
class WangSmoothLayer : public Layer<Dtype> {
 public:
  
  explicit WangSmoothLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_WANGSMOOTH;
  }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented (non-differentiable function)
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    // Do nothing
    // NOT_IMPLEMENTED;
  }
  int num_, channels_, height_, width_;
  bool out_max_val_;
  size_t top_k_;
  //Dtype * mask_around;
};


template <typename Dtype>
class WangBigfeaLayer : public Layer<Dtype> {
 public:
  
  explicit WangBigfeaLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_WANGBIGFEA;
  }
  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented (non-differentiable function)
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    // Do nothing
    // NOT_IMPLEMENTED;

  int num_, channels_, height_, width_, channels_sec;
  bool out_max_val_;
  size_t top_k_;
  Blob<Dtype> bottom_diff_wang;
  vector<vector<std::pair<int,int> > > bigregion_size;
};



template <typename Dtype>
class WangRnntestLayer : public Layer<Dtype> {
 public:
  
  explicit WangRnntestLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_WANGRNNTEST;
  }
  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 4; }

 protected:
  
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented (non-differentiable function)
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  void WangUpdate();
  void WangSnapshot(const int fealen);
  int findrelation(int c0, vector<vector<int> > treestruct, int tree_ii);
  int num_, channels_, height_, width_;
  bool out_max_val_;
  size_t top_k_;
  //int max_labels;

  Blob<Dtype> tmp_bottom;
  Blob<Dtype> tmp_top;
  Blob<Dtype> tmp_top_jihuo_dao;

  Blob<Dtype> error_bottom;
  Blob<Dtype> error_top;
  Blob<Dtype> error_top_jihuo_dao;

  Blob<Dtype> error_bottom_tmp;
  Blob<Dtype> error_top_tmp;




  Blob<Dtype> bottom_new;
  Blob<Dtype> rnn_weight;
  Blob<Dtype> rnn_weight2;
  Blob<Dtype> rnn_weight0;
  Blob<Dtype> bottom_diff_wang;
  Blob<Dtype> top_diff_wang;

  Blob<Dtype> rnn_weight_history;
  Blob<Dtype> rnn_weight2_history;


  Blob<Dtype> relation_label;
  vector<vector<int> > bigkeywords;
  vector<vector<vector<int> > > bigerror_lib;

  std::string addr;
  
};




template <typename Dtype>
class WangRnnLayer : public Layer<Dtype> {
 public:
  
  explicit WangRnnLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_WANGRNN;
  }
  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented (non-differentiable function)
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  void WangUpdate();
  void WangSnapshot(const int fealen);
  int findrelation(int c0, vector<vector<int> > treestruct, int tree_ii);
  int num_, channels_, height_, width_;
  bool out_max_val_;
  size_t top_k_;
  //int max_labels;

  Blob<Dtype> tmp_bottom;
  Blob<Dtype> tmp_top;
  Blob<Dtype> tmp_top_jihuo_dao;

  Blob<Dtype> error_bottom;
  Blob<Dtype> error_top;
  Blob<Dtype> error_top_jihuo_dao;

  Blob<Dtype> error_bottom_tmp;
  Blob<Dtype> error_top_tmp;




  Blob<Dtype> bottom_new;
  Blob<Dtype> rnn_weight;
  Blob<Dtype> rnn_weight2;
  Blob<Dtype> rnn_weight0;
  Blob<Dtype> bottom_diff_wang;
  Blob<Dtype> top_diff_wang;

  Blob<Dtype> rnn_weight_history;
  Blob<Dtype> rnn_weight2_history;


  Blob<Dtype> relation_label;
  vector<vector<int> > bigkeywords;
  vector<vector<vector<int> > > bigerror_lib;

  std::string addr;
  
};




template <typename Dtype>
class WangconLayer : public Layer<Dtype> {
 public:
  explicit WangconLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_WANGCON;
  }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> col_bob_;
  int count_;
  int num_;
  int channels_;
  int height_;
  int width_;
  int concat_dim_;
};



template <typename Dtype>
class WangbrokenLayer : public Layer<Dtype> {
 public:
  explicit WangbrokenLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_WANGBROKEN;
  }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  //virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  //    const vector<Blob<Dtype>*>& top);

 
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> col_bob_;
  int count_;
  int num_;
  int channels_;
  int height_;
  int width_;
  int concat_dim_;
};


}  // namespace caffe

#endif  // CAFFE_COMMON_LAYERS_HPP_
