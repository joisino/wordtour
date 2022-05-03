#include<vector>
#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<cmath>

int n, D;

std::vector<std::vector<double>> vs;

double d(int i, int j){
  double res = 0;
  for(int k = 0; k < D; k++){
    double r = vs[i][k] - vs[j][k];
    res += r * r;
  }
  return sqrt(res);
}

double score(std::vector<int> tour){
  double res = 0;
  for(int i = 0; i < n-1; i++){
    res += d(tour[i], tour[i+1]);
  }
  return res;
}

int main(int argc, char **argv){

  if(argc != 3){
    std::cerr << "Usage: " + std::string(argv[0]) + " [embedding file path] [#words]" << std::endl;
    exit(1);
  }

  std::ifstream ifs(argv[1]);
  n = std::stoi(argv[2]);

  for(int i = 0; i < n; i++){
    std::string line;
    std::getline(ifs, line);
    std::stringstream ss(line);
    std::string word;
    ss >> word;
    std::vector<double> v(0);
    while(!ss.eof()){
      double buf;
      ss >> buf;
      v.push_back(buf);
    }
    D = v.size();
    vs.push_back(v);
  }

  std::cout << "NAME: wordtour" << std::endl;
  std::cout << "TYPE: TSP" << std::endl;
  std::cout << "DIMENSION: " << n << std::endl;
  std::cout << "EDGE_WEIGHT_TYPE: EXPLICIT" << std::endl;
  std::cout << "EDGE_WEIGHT_FORMAT: UPPER_ROW" << std::endl;
  std::cout << "EDGE_WEIGHT_SECTION" << std::endl;
  for(int i = 0; i < n; i++){
    for(int j = i + 1; j < n; j++){
      std::cout << int(d(i, j) * 1000) << " ";
    }
    std::cout << std::endl;
  }
  
  return 0;
}
	