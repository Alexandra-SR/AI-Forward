#include <bits/stdc++.h>
#include "Eigen/Core"
#include "Eigen/Dense"

using namespace std;

typedef vector<double> vT;
typedef vector<int> vi;

void parse(string buffer, int i, Eigen::MatrixXd& df)
{
    stringstream buff(buffer);
    string att; 
    int j = 0;
    while (getline(buff, att, ','))
    {   
        // cout << stod(att) << " (" << i << ", " << j << ") ";
        df(i, j) = stod(att);
        j++;
    }
}

Eigen::MatrixXd readCSV(std::string file, int rows, int cols) 
{
    auto in = ifstream(file);
    string buff;
    Eigen::MatrixXd df(rows, cols);
    int i = 0;
    while (std::getline(in, buff) && i < rows)
    {
        parse(buff, i, df);
        i++;
    }
    return df;
}


auto sigmoid(double net)        // vector salidas/netas
{
    return 1/(1 + exp(-1*net));
}

Eigen::MatrixXd generate_matrix(int ii, int jj)
{
    Eigen::MatrixXd mat(ii, jj);
    for (int i = 0; i < ii; ++i)
    {
        for (int j = 0; j < jj; ++j)
        {
            mat(i, j) = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        }
    }
    return mat;
}

template<typename fAct>
struct NN
{
    vector<Eigen::MatrixXd> layers;
    fAct FF;

    NN(int i, int nH, vector<int> nph, int O ,fAct f): FF{f}
    {
        vector<int> l_size = {i};
        for (const auto& n : nph)
            l_size.push_back(n);
        l_size.push_back(O);        // 3 3 2 2 2 -> layers_dim: 3*3 3*2 2*2 2*2

        layers = vector<Eigen::MatrixXd>(nH + 1);
        for (int i = 1; i < size(l_size); ++i)
        {
            auto ii = l_size[i-1] + 1, jj = l_size[i];      // i + 1 x bias
            layers[i-1] = generate_matrix(ii, jj);
            // cout << "\t" << ii << " " << jj << endl;
            // cout << layers[i-1] << endl;
        }
    }

    void run(Eigen::MatrixXd feature_vector)
    {
        this->forward(feature_vector);
    }

    Eigen::MatrixXd forward(Eigen::MatrixXd fv)
    {
        Eigen::MatrixXd output = Eigen::MatrixXd(1, fv.size());
        for (int i = 0; i < fv.size(); ++i)
            output(i) = fv(i);
        
        for (int i = 0; i < size(this->layers); ++i)
        {
            cout << "\t\tCAPA " << i << endl;
            cout << "input\n\t" << output << endl;
            cout << "layer\n\t" << layers[i] << endl;

            Eigen::MatrixXd neta = output*layers[i];
            cout << "neta\n\t" << neta << " " << endl;

            auto n_size = neta.size() + 1;
            output.resize(1, n_size);
            for (int pos = 0; pos < neta.size(); ++pos)
                output(pos) = FF(neta(pos));
            output(n_size-1) = 1;
            // cout << "output\n\t" << output << endl;
        }
        cout << "\tFORWARD: " << output << endl;
        return output;
    }
};

int main()
{
#ifndef TEST
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
#endif
    
    int I, NH, O;
    cin >> I >> NH;
    vector<int> nph(NH);
    // Eigen::MatrixXd nph(1, NH);
    for (int i = 0; i < NH; ++i)
        cin >> nph[i];
    cin >> O;
    // cout << nph << endl;

    auto nn = NN(I, NH, nph, O, sigmoid);
    cout << "Initialized\n";

    Eigen::MatrixXd fv(1, I + 1);
    for (int i = 0; i < I; ++i)
        cin >> fv(i);
    fv(I) = 1;

    // cout << fv.size() << endl;
    // cout << fv << endl;
    nn.run(fv);

    // auto csv = readCSV("test_data.csv", 4, 10);
    // cout << csv;
    
    // Eigen::MatrixXd m(2,2);
    // m(0,0) = 3;
    // m(1,0) = 2.5;
    // m(0,1) = -1;
    // m(1,1) = m(1,0) + m(0,1);
    // cout << m << std::endl;
    // parse("30,64,1,1", 0);
    // parse("30.5,64.3423,1.457,1.9894", 1);
    // cout << readCSV("test_data.csv", 10, 4) << '\n';
    cout << "Done\n";
    return 0;
}
