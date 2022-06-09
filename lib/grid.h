#ifndef __grid_h__
#define __grid_h__
#include "common.h"

class Grid
{
protected:
    int num_grid_points_;
    real x_min_;
    real x_max_;
    real dx_;
    real dk_;

    vector_real fourier_space_;

public:
    Grid()
    {
    }

    virtual void init(int num_grid_points,
              real x_min,
              real x_max)

    {
        x_min_ = x_min;
        x_max_ = x_max;
        num_grid_points_ = num_grid_points;
        dx_ = (x_max_ - x_min_) / (num_grid_points_ - 1);
        dk_ = 2 * m_pi / (x_max_ - x_min_);

        fourier_space_.resize(num_grid_points_);

        for (int i = 0; i < num_grid_points_; ++i)
        {
            fourier_space_[i] = i < num_grid_points_ / 2 ? i * dk_ : (i - num_grid_points_) * dk_;
        }
    }

    real XMin()
    {
        return x_min_;
    }

    real XMax()
    {
        return x_max_;
    }

    real Dx()
    {
        return dx_;
    }

    real Dk()
    {
        return dk_;
    }

    real Coord(int i)
    {
        return x_min_ + (i - 1) * dx_;
    }

    int NumOfGridPoints()
    {
        return num_grid_points_;
    }

    real FourierSpace(int i)
    {
        return fourier_space_[i];
    }
};

#endif