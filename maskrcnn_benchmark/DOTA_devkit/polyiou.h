//
// Created by dingjian on 18-2-3.
//

#ifndef POLYIOU_POLYIOU_H
#define POLYIOU_POLYIOU_H

#include <vector>

typedef struct {
            double iou;
            double inter1;
            double inter2;
} iou_test;

iou_test iou_poly(std::vector<double> p, std::vector<double> q);
#endif //POLYIOU_POLYIOU_H
