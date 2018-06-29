//
// Created by Guofeng on 2018/06/29..
//

#include "featuredb.h"



FeatureDB::FeatureDB(const std::string path, float thres)
{
    dbfile = path + "feature.db";
    threshold = thres;

    load_feature();
}
FeatureDB::~FeatureDB()
{
    save_feature();
}

void FeatureDB::add_feature(const std::string name, std::vector<float> feature)
{
    features.insert(map<string, vector<float>>::value_type(name, feature));
    save_feature();
}
void FeatureDB::del_feature(const std::string name)
{
    features.erase(name);
    save_feature();
}
std::string FeatureDB::find_name(std::vector<float> feature)
{
    map<string,vector<float>>::iterator it;
    float largest = 0, similar;
    std::string name = "";

    it = features.begin();
    while(it != features.end()) {
        similar = cal_similar(it->second, feature);
        if (similar > largest) {
            largest = similar;
            name = it->first;
            cout<<it->first<<endl;
        }

        it ++;
    }

    if (largest < threshold)
        return name;
    else
        return "";
}

void FeatureDB::save_feature()
{
    ofstream of(dbfile);

    map<string,vector<float>>::iterator it;

    it = features.begin();
    while(it != features.end()) {
        of << it->first << " ";
	cout << it->first << " ";

	for (int i = 0; i != it->second.size(); ++i) {
            of << it->second[i] << " ";
            cout << it->second[i] << " ";
        }

	of << endl;
        cout << endl;
        it ++;
    }

}
void FeatureDB::load_feature()
{}

float FeatureDB::cal_similar(std::vector<float>& v1, std::vector<float>& v2)
{
   assert(v1.size() == v2.size());
   double ret = 0.0, mod1 = 0.0, mod2 = 0.0;
   for (int i = 0; i != v1.size(); ++i)
   {
      ret += v1[i] * v2[i];
      mod1 += v1[i] * v1[i];
      mod2 += v2[i] * v2[i];
   }
   return (ret / sqrt(mod1) / sqrt(mod2) + 1) / 2.0;
}
