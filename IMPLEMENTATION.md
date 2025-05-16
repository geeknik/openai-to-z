# Cost-Effective Implementation of Amazon Archaeological Discovery Tool

This document outlines the cost-optimization strategies implemented in our approach to the OpenAI to Z Challenge. We've designed a system that balances functionality with minimal resource usage.

## Cost Optimization Strategies

### 1. Data Management

- **Free Data Sources**: Exclusively uses open-source satellite imagery (Sentinel), LiDAR (OpenTopography), and global elevation data (SRTM).
- **Sampling & Caching**:
  - Downloads data at lower resolution for initial analysis (configurable via `SAMPLE_RESOLUTION`)
  - Implements intelligent caching system with customizable expiration times to prevent redundant downloads
  - Stores processed results to minimize recomputation

### 2. API Usage Optimization

- **Tiered Model Approach**:
  - Uses cheaper `o3-mini` model for initial feature screening
  - Only escalates to more expensive `gpt-4` model for ambiguous cases (0.3-0.7 confidence)
  - Implements cache for all AI responses to prevent duplicate API calls
  - Batches requests where possible to reduce total API calls

- **Prompt Engineering**:
  - Carefully designed prompts minimize token usage while maintaining accuracy
  - Returns structured data that's easy to parse without additional processing

### 3. Computational Resources

- **Local Processing Pipeline**:
  - Core image processing and feature detection use efficient OpenCV and scikit-image algorithms
  - These run locally to avoid cloud compute costs
  - Optimized algorithms reduce required memory and processing power

- **Database Strategy**:
  - Uses SQLite with SpatiaLite for development/small datasets
  - Option to scale to PostgreSQL/PostGIS for production only when needed

- **Lightweight Visualization**:
  - Employs Folium (wrapper for Leaflet.js) instead of paid solutions like Mapbox
  - Generates static exports when possible to avoid hosting costs

### 4. Development Approach

- **Incremental Region Analysis**:
  - Starts with small, targeted regions to validate approach before scaling
  - Uses configurable region bounds to test in areas with known archaeological sites

- **Validation Pipeline**:
  - Two-source confirmation rule ensures high-confidence discoveries
  - Reduces false positives that could lead to wasted research resources

## Implementation Details

### Caching System

The robust caching system (`utils/cache.py`) provides:
- Function result caching with parameter-based invalidation
- Selective cache clearing by prefix
- Configurable expiration times
- Support for both pickle and JSON serialization

### OpenAI API Usage

The AI utilities (`utils/ai.py`) optimize API usage through:
- Response caching with model and prompt-based keys
- Exponential backoff for rate limit handling
- Two-stage classification (cheap model → expensive model only when needed)
- Prompt templates optimized for token efficiency

### Data Processing

Data fetching and processing (`preprocessing/data_fetcher.py`) implements:
- Incremental downloading of spatial data
- Free API access via OpenTopography and Google Earth Engine
- Automatic downsampling to reduce data size

### Feature Detection

The feature detection system (`analysis/feature_detector.py`) balances accuracy and efficiency:
- Classic computer vision algorithms first (lower computational cost)
- AI confirmation only for promising features
- Feature merging to reduce redundant detections

## Cost Comparison

| Resource | Traditional Approach | Our Implementation | Savings |
|----------|----------------------|-------------------|---------|
| Satellite Imagery | Commercial providers ($1000s/region) | Free Sentinel/Landsat | 95-100% |
| LiDAR Data | Custom flights ($10,000s) | Free OpenTopography | 95-100% |
| Cloud Storage | Full resolution (TBs, $100s/month) | Sampled resolution (GBs, $0-10/month) | 90-100% |
| AI API Costs | Entire pipeline through AI ($100s/day) | Targeted AI usage with caching ($1-10/day) | 90-95% |
| Compute | 24/7 cloud instances ($100s/month) | On-demand processing with local option | 80-95% |

## Scaling Considerations

The system is designed for gradual scaling:

1. **Development Phase**: Run entirely locally with SQLite and small regions
2. **Validation Phase**: Use minimal cloud resources for specific target regions
3. **Production Phase**: Scale to cloud infrastructure only for the full Amazon basin analysis

This tiered approach ensures resources are only allocated when absolutely necessary.

## Conclusion

Our implementation demonstrates that cutting-edge archaeological discovery can be performed cost-effectively by leveraging open data sources, efficient algorithms, and strategic use of AI capabilities. The system is designed to scale with research needs while maintaining minimal operating costs. 