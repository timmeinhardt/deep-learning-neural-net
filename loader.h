#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <memory>
#include <assert.h>

#include <machine/endian.h>

using namespace std;

class Loader final {
public:
  // Default constructor
  Loader();

  // Destructor
  ~Loader();

  // Accessors
  size_t GetImageWidth() const;
  size_t GetImageHeight() const;
  size_t GetImageCount() const;
  size_t GetImageSize() const;
  const float* GetImageData() const;
  const uint8_t* GetCategoryData() const;

  void Print();

  int Parse(const char*, const char*);

private:  
  size_t m_count; // The total number of images

  size_t m_width; // Dimension of the image data
  size_t m_height;
  size_t m_imageSize;

  float* m_buffer; // The entire buffers that stores both the image data and the category data
  float* m_imageBuffer;
  static const int c_categoryCount = 10;
  uint8_t* m_categoryBuffer; // 1-of-N label of the image data (N = 10) 

  void Initialize(const size_t, const size_t, const size_t);
};