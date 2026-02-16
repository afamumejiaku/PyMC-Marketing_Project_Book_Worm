# 📊 Marketing Mix Model - Project Book-Worm

A Bayesian Marketing Mix Modeling (MMM) implementation using [PyMC-Marketing](https://github.com/pymc-labs/pymc-marketing) to measure marketing channel effectiveness and optimize budget allocation for a Book-worm dataset.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyMC-Marketing](https://img.shields.io/badge/PyMC--Marketing-0.18.0-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📖 Overview

This project implements a **Marketing Mix Model (MMM)** to:

- 📈 **Measure channel effectiveness** - Quantify how each marketing channel contributes to subscriptions
- 💰 **Calculate ROI** - Determine return on investment for every dollar spent
- 🎯 **Optimize budget allocation** - Provide data-driven recommendations for budget reallocation
- 📊 **Account for marketing dynamics** - Model carryover effects and diminishing returns


### Key Components

| Component | Description | Implementation |
|-----------|-------------|----------------|
| **Adstock** | Models carryover effect (ads today → conversions later) | Geometric decay with α parameter |
| **Saturation** | Captures diminishing returns | Logistic function |
| **Seasonality** | Yearly patterns in subscriptions | Fourier series (2 terms) |
| **Controls** | Non-media factors affecting KPI | Linear regression terms |

## 📂 Data Description

### Input Variables

| Category | Variable | Description |
|----------|----------|-------------|
| **KPI** | `Accounts Subscriptions` | Weekly new subscriptions (target) |
| **Offline Media** | `TV_GRP` | TV Gross Rating Points |
| **Digital Display** | `Google_Display_Impressions` | Google Display Network impressions |
| **Social** | `Meta_Impressions` | Facebook/Instagram impressions |
| **Influencer** | `Influencers_Views` | Influencer content views |
| **Search** | `Google_Generic_Paid_Search_Impressions` | Non-branded search impressions |
| **Search** | `Google_Brand_Paid_Search_Clicks` | Branded search clicks |
| **Video** | `YouTube_Impressions` | YouTube ad impressions |
| **Controls** | `Promotion` | Discount percentage |
| **Controls** | `Dates_School_Holidays` | School holiday days (0-7) |
| **Controls** | `Competitors Promotion` | Competitor discount activity |

### Data Structure

- **Granularity**: Weekly (157 weeks)
- **Period**: January 2022 - December 2024
- **Level**: National aggregate

## 🚀 Quick Start

### Requirements

```txt
pymc-marketing>=0.18.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
openpyxl>=3.1.0
arviz>=0.15.0
```

### Running the Model

```bash
# Basic execution
python bookworm_mmm.py

# Or import as module
python -c "from bookworm_mmm import main; main()"
```

## 📊 Results

### Grapical View
![Result](outputs/results.png)

### ROI Ranking

| Rank | Channel | ROI | CPA |
|------|---------|-----|-----|
| 1 | Influencers | 0.2985 | $3.35 |
| 2 | Google Generic Search | 0.0798 | $12.53 |
| 3 | Google Brand Search | 0.0743 | $13.45 |
| 4 | YouTube | 0.0628 | $15.91 |
| 5 | Google Display | 0.0264 | $37.95 |
| 6 | TV | 0.0239 | $41.89 |
| 7 | Meta | 0.0173 | $57.80 |

### Adstock Parameters

| Channel | Alpha (α) | Half-Life |
|---------|-----------|-----------|
| TV | 0.589 | 1.3 weeks |
| Influencers | 0.316 | 0.6 weeks |
| Meta | 0.244 | 0.5 weeks |
| Google Display | 0.242 | 0.5 weeks |
| YouTube | 0.186 | 0.4 weeks |
| Generic Search | 0.187 | 0.4 weeks |
| Brand Search | 0.181 | 0.4 weeks |

### Budget Recommendations

| Channel | Current Status | Recommendation |
|---------|---------------|----------------|
| 🟢 Influencers | High ROI | **INCREASE** budget 20-30% |
| 🟡 Google Search | Average ROI | **MAINTAIN** current level |
| 🟡 YouTube | Average ROI | **MAINTAIN** current level |
| 🔴 Google Display | Low ROI | **DECREASE** and reallocate |
| 🔴 TV | Low ROI | **DECREASE** and reallocate |
| 🔴 Meta | Low ROI | **DECREASE** and reallocate |

## 📁 Project Structure

```
PyMC-Marketing_Project_Book_Worm/
├── 📄 README.md                   # This file
├── 📄 requirements.txt            # Python dependencies
├── 📄 bookworm_mmm.py             # Main implementation
├── ProjectInputData.xlsx          # Media activity & controls
├── MediaSpend.xlsx                # Spend by channel
├── 📂 outputs/
│   ├── mmm_results.png            # Visualization dashboard
│   ├── channel_performance.csv    # Detailed metrics
│   └── executive_summary.txt      # Text report
└──  exploratory_analysis.ipynb # EDA notebook
```

## ⚙️ Configuration

Modify the `Config` class in `bookworm_mmm.py`:

```python
class Config:
    # MCMC parameters (increase for production)
    MCMC_DRAWS = 1000      # Posterior samples
    MCMC_TUNE = 1000       # Tuning iterations
    MCMC_CHAINS = 4        # Parallel chains
    TARGET_ACCEPT = 0.9    # Acceptance rate
    
    # Model parameters
    ADSTOCK_MAX_LAG = 8    # Max carryover weeks
    YEARLY_SEASONALITY = 2 # Fourier terms
```

### Production Settings

For robust inference, use:
- `MCMC_DRAWS`: 2000-4000
- `MCMC_TUNE`: 2000
- `MCMC_CHAINS`: 4 (minimum)
- Check `r_hat < 1.01` for convergence

## 🔬 Methodology

### Adstock Transformation

Models the delayed effect of advertising:

```
adstock_t = activity_t + α × adstock_{t-1}
```

Where α ∈ (0,1) is the retention rate. Higher α = longer-lasting effect.

**Half-life calculation:**
```
half_life = log(0.5) / log(α)
```

### Saturation Function

Captures diminishing returns using logistic function:

```
saturation(x) = 1 / (1 + exp(-λ × (x - μ)))
```

### Bayesian Framework

- **Prior distributions**: Weakly informative priors
- **Inference**: NUTS sampler (No U-Turn Sampler)
- **Posterior**: Full uncertainty quantification

## 📈 Interpreting Results

### ROI (Return on Investment)
```
ROI = Total Subscriptions Driven / Total Spend
```
- Higher ROI = more efficient channel
- Compare relative to average ROI

### CPA (Cost Per Acquisition)
```
CPA = Total Spend / Total Subscriptions Driven
```
- Lower CPA = more cost-effective
- Target channels with lowest CPA for growth

### Contribution Share vs Spend Share
- If Contribution % > Spend % → Efficient (above diagonal)
- If Contribution % < Spend % → Inefficient (below diagonal)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📚 References

- [PyMC-Marketing Documentation](https://www.pymc-marketing.io/)
- [PyMC-Marketing GitHub](https://github.com/pymc-labs/pymc-marketing)
- [Bayesian Methods for Media Mix Modeling](https://research.google/pubs/pub46001/)
- [Jin, Y., et al. (2017). Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46001.pdf)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 

## 👤 Author

**Afamefuna Umejiaku**
For Employment Purpose
---

<p align="center">
  <i>Built with ❤️ using PyMC-Marketing</i>
</p>
