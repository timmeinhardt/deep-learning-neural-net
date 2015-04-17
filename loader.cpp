#include "loader.h"

using namespace std;

Loader::Loader()
  : m_count(0),
  m_width(0),
  m_height(0),
  m_imageSize(0),
  m_buffer(nullptr),
  m_imageBuffer(nullptr),
  m_categoryBuffer(nullptr)
{
}

Loader::~Loader()
  {
    if (m_buffer) free(m_buffer);
    if (m_categoryBuffer) free(m_categoryBuffer);
  }

void Loader::Print()
  {
    for (size_t n = 0; n < m_count; ++n)
    {
      const float* imageBuffer = &m_imageBuffer[n * m_imageSize];
      for (size_t j = 0; j < m_height; ++j)
      {
        for (size_t i = 0; i < m_width; ++i)
        {
          printf("%3d ", (uint8_t)imageBuffer[j * m_width + i]);
        }
        printf("\n");
      }

      printf("\n [%zd] ===> cat(%u)\n\n", n, m_categoryBuffer[n]);
    }
  }

size_t Loader::GetImageWidth() const
  {
    return m_width;
  }

size_t Loader::GetImageHeight() const
  {
    return m_height;
  }

size_t Loader::GetImageCount() const
  {
    return m_count;
  }

size_t Loader::GetImageSize() const
  {
    return m_imageSize;
  }

const float* Loader::GetImageData() const
  {
    return m_imageBuffer;
  }

const uint8_t* Loader::GetCategoryData() const
  {
    return m_categoryBuffer;
  }

int Loader::Parse(DataSet& set, const char* imageFile, const char* labelFile)
  {
    FILE* fimg = nullptr;
    fimg = fopen(imageFile, "rb");
    if (fimg == NULL)
    {
      printf("Failed to open %s for reading\n", imageFile);
      return 1;
    }
    
    FILE* flabel = nullptr;
    flabel = fopen(labelFile, "rb");
    if (flabel == NULL)
    {
      printf("Failed to open %s for reading\n", labelFile);
      return 1;
    }
    shared_ptr<void> autofimg(nullptr, [fimg, flabel](void*) {
      if (fimg) fclose(fimg);
      if (flabel) fclose(flabel);
    });

    uint32_t value;

    // Read magic number
    assert(!feof(fimg));
    fread(&value, sizeof(uint32_t), 1, fimg);
    printf("Image Magic        :%0X(%u)\n", ntohl(value), ntohl(value));
    //assert(ntohl(value) == 0x00000803);

    // Read count
    assert(!feof(fimg));
    fread(&value, sizeof(uint32_t), 1, fimg);
    const uint32_t count = ntohl(value);
    printf("Image Count        :%0X(%u)\n", count, count);
    assert(count > 0);

    // Read rows
    assert(!feof(fimg));
    fread(&value, sizeof(uint32_t), 1, fimg);
    const uint32_t rows = ntohl(value);
    printf("Image Rows         :%0X(%u)\n", rows, rows);
    assert(rows > 0);

    // Read cols
    assert(!feof(fimg));
    fread(&value, sizeof(uint32_t), 1, fimg);
    const uint32_t cols = ntohl(value);
    printf("Image Columns      :%0X(%u)\n", cols, cols);
    assert(cols > 0);

    // Read magic number (label)
    assert(!feof(flabel));
    fread(&value, sizeof(uint32_t), 1, flabel);
    printf("Label Magic        :%0X(%u)\n", ntohl(value), ntohl(value));
    assert(ntohl(value) == 0x00000801);

    // Read label count
    assert(!feof(flabel));
    fread(&value, sizeof(uint32_t), 1, flabel);
    printf("Label Count        :%0X(%u)\n", ntohl(value), ntohl(value));
    // The count of the labels needs to match the count of the image data
    assert(ntohl(value) == count);

    Initialize(cols, rows, count);

    size_t counter = 0;
    while (!feof(fimg) && !feof(flabel) && counter < m_count)
    {
      //float* imageBuffer = &m_imageBuffer[counter * m_imageSize];
      gsl_vector* v = gsl_vector_alloc(m_imageSize);

      for (size_t i = 0; i < m_height * m_width; ++i)
      {
        uint8_t pixel;
        fread(&pixel, sizeof(uint8_t), 1, fimg);
        //imageBuffer[j * m_width + i] = pixel;
        gsl_vector_set(v, i, pixel);
      }

      uint8_t cat;
      fread(&cat, sizeof(uint8_t), 1, flabel);
      // assert(cat >= 0 && cat < c_categoryCount);
      // m_categoryBuffer[counter] = cat;
      set.push_back(pair<gsl_vector*, int>(v, cat));

      ++counter;
    }

    return 0;
  }

void Loader::Initialize(const size_t width, const size_t height, const size_t count)
  {
    m_width = width;
    m_height = height;
    m_imageSize = m_width * m_height;
    m_count = count;

    m_buffer = (float*)malloc(m_count * m_imageSize * sizeof(float));
    m_imageBuffer = m_buffer;
    m_categoryBuffer = (uint8_t*)malloc(m_count * sizeof(uint8_t));
  }
