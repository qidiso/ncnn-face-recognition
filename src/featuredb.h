#pragma once
#ifndef FEATUREDB_H_
#define FEATUREDB_H_

#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <map>
#include <math.h>
#include <assert.h>

using namespace std;


class FeatureDB {
public:
    FeatureDB(const std::string path, float thres);
    ~FeatureDB();
    void add_feature(const std::string name, std::vector<float> feature);
    void del_feature(const std::string name);
    std::string find_name(std::vector<float> feature);
private:
    void save_feature();
    void load_feature();
    float cal_similar(std::vector<float>& v1, std::vector<float>& v2);

private:
    std::string dbfile;
    map<string, vector<float>> features;
    float threshold;
};

#endif /* FEATUREDB_H_ */
