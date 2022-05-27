
#include <iostream>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <vector>
#include <cmath>

using namespace boost::numeric::ublas;

class NeuralNet{
    private:
        std::vector<matrix<double>> vectorcito;
        int input_size = 2;
        int hidden_size = 3;
        int output_size = 1;

        //TODO: NOT LIKE THIS
        int weights_size = 2;

        std::vector<float> weights;
    public:
        NeuralNet();
        void ForwardPro();
        float sigmoid_f(matrix<double> x);
};

NeuralNet::NeuralNet(){
    for(int i = 0; i < weights_size; i++){
        float num = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        weights.push_back(num);
    }
}

float NeuralNet::sigmoid_f(matrix<double> x){
    for (int i = 0; i < x.size1(); i++){
        for(int j = 0; j < x.size2(); j++){
            x(i, j) = 1/exp(x(i, j));
        }
    }
    return ;
}


void NeuralNet::ForwardPro(){
    int count = 0;
    for(auto &c : vectorcito){
        
    }
}





int main(){

    
/*
    matrix<double> m (3, 3);
    for (unsigned i = 0; i < m.size1 (); ++ i)
        for (unsigned j = 0; j < m.size2 (); ++ j)
            m (i, j) = 3 * i + j;
    std::cout << m << std::endl;
    return 0;
*/
}
