#include <bits/stdc++.h>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <sys/types.h>
#include <sys/stat.h>

using namespace std;
using namespace boost::numeric::ublas;



random_device rd;
seed_seq ssq{rd()};
default_random_engine eng{rd()};    
uniform_real_distribution<double> dist(-9e-1, 9e-1);


int RNG(int first, int last)
{
    return first + rand() % (last + 1 - first);
}


string DATA_PATH="./data/", MODEL_PATH="";
ofstream ERROR_STREAM;

// activation functions
void sigmoid_(matrix<double>& mt)
{
    for (int i = 0; i < mt.size1(); ++i)
        for (int j = 0; j < mt.size2(); ++j)
            mt(i,j) = 1/(1 + exp(-1*mt(i,j)));
        
}
void tanh_(matrix<double>& mt)
{
    for (int i = 0; i < mt.size1(); ++i)
        for (int j = 0; j < mt.size2(); ++j)
            mt(i, j) = 2/(1+ exp(-2*mt(i, j))) - 1;
}
void relu_(matrix<double>& mt)
{
    for (int i = 0; i < mt.size1(); ++i)
        for (int j = 0; j < mt.size2(); ++j)
            mt(i, j) = (mt(i, j) > 0) ? mt(i, j) : 0.1 * mt(i, j);
}
void soft_max_(matrix<double>& mt)
{
    double sum = 0;
    for (int i = 0; i < mt.size1(); ++i)
    {
        sum = 0;
        for (int k = 0; k < mt.size2(); ++k)
            sum += exp(mt(i, k));
        for (int j = 0; j < mt.size2(); ++j)
            mt(i, j) = exp(mt(i, j))/sum;
    }
}

// derivates
matrix<double> der_cross_entropy_(matrix<double> mt, matrix<double> y)
{
    for (int pos = 0; pos < mt.size1(); ++pos)
        mt(pos) = -1 * (y(pos) * (1/mt(pos)) + (1-y(pos))*(1/(1-mt(pos))));
    return mt;
}
matrix<double> der_soft_max_(matrix<double> mt)
{
    double sum = 0;
    for (int i = 0; i < mt.size1(); ++i)
    {
        sum = 0;
        for (int k = 0; k < mt.size2(); ++k)
            sum += exp(mt(i, k));
        for (int j = 0; j < mt.size2(); ++j)
            mt(i, j) = (exp(mt(i, j))*(sum-mt(i, j)))/pow(sum, 2);
    }
        
    return mt;
}
matrix<double> der_tanh_(matrix<double> mt)
{
    for (int pos = 0; pos < mt.size1(); ++pos)
        mt(pos) = 1.0 - pow(2/(1+ exp(-2*mt(pos)))-1, 2);
    return mt;
}
matrix<double> der_sigm_(matrix<double> mt)
{
    for (int pos = 0; pos < mt.size1(); ++pos)
        mt(pos) = mt(pos)*(1-mt(pos));
    return mt;
}
matrix<double> der_relu_(matrix<double> mt)
{   // Leaky RELU
    for (int i = 0; i < mt.size1(); ++i)
        for (int j = 0; j < mt.size2(); ++j)
            mt(i, j) = (mt(i, j) > 0) ? 1 : 0.1;
    return mt;
}

// utils
matrix<double> generate_matrix(int ii, int jj)
{
    matrix<double> mat(ii, jj);
    for (int i = 0; i < ii; ++i)
    {
        for (int j = 0; j < jj; ++j)
        {
            mat(i, j) = dist(eng);
        }
    }
    return mat;
}
template<typename T>
pair<int, int> get_shape(matrix<T> m)
{ 
    return make_pair(m.size1(), m.size2()); 
}
template<typename T>
ostream& operator<<(ostream& os, pair<T, T> p)
{
    os << "(" << p.first << ", " << p.second << ")";
    return os;
}
template<typename T>
ostream& operator<<(ostream& os, matrix<T> m)
{
    auto [r, c] = get_shape(m);
    for (unsigned i = 0; i < r; ++ i)
    {
        for (unsigned j = 0; j < c; ++ j)
            os << m(i, j) << " ";
        os << endl;
    }
    return os;
}
template<typename T>
void show_(matrix<T> m)
{
    auto [r, c] = get_shape(m);
    cout << make_pair(r, c) << endl;
    for (unsigned i = 0; i < r; ++ i)
    {
        for (unsigned j = 0; j < c; ++ j)
            cout << setw(15) << m(i, j);
        cout << endl;
    }   
}
template<typename T>
void export_(string file, matrix<T> mat)
{
    auto of = ofstream(file.c_str(), ios::trunc);
    of << mat << endl;
    of.close();
}
template<typename T>
void import_(string file, matrix<T>& mat)
{
    auto is = ifstream(file.c_str());
    auto [r, c] = get_shape(mat);
    string line, value;
    int i = 0, j = 0;
    while (getline(is, line))
    {
        stringstream ss(line);
        j = 0;
        while (ss >> value)
            mat(i, j) = stod(value), j++;
        i++;
    }
    is.close();
}
void parse(const string& buffer, int i, matrix<double>& df) 
{
    stringstream buff(buffer);
    string att; 
    int j = 0;
    while (getline(buff, att, ',')) {   
        df(i, j) = stod(att);
        j++;
    }
}
matrix<double> read_csv(const string& file, int rows)
{
    auto in = ifstream(file);
    string buff;

    getline(in, buff);  
    stringstream ss(buff);
    int cols = 0;
    while (getline(ss, buff, ','))
        cols++;
    
    // fill matrix
    matrix<double> df(rows, cols);
    int i = 0;
    while (getline(in, buff) && i < rows) {
        parse(buff, i, df);
        i++;
    }

    return df;
}

// maps
map<string, void(*)(matrix<double>&)> activation_function = {
    {"sigmoid", sigmoid_},
    {"relu", relu_},
    {"tanh", tanh_},
    {"soft-max", soft_max_}
};
map<string, matrix<double>(*)(matrix<double>, matrix<double>)> error_derivatives = {
    {"cross-entropy", der_cross_entropy_}
};
map<string, matrix<double>(*)(matrix<double>)> activ_derivatives = {
    {"soft-max", der_soft_max_},
    {"sigmoid", der_sigm_}, 
    {"tanh", der_tanh_}, 
    {"relu", der_relu_}, 
};

struct MLP
{
    std::map<string, matrix<double>> data;
    std::vector<int> npl;    
    std::vector<matrix<double>> W, W_change;
    std::vector<matrix<double>> delta, S, bias;
    std::vector<string> activ;
    string error_func = "cross-entropy";
    double learning_rate = 1e-6, lambda = 1, betha1=0.9, betha2=0.999, eps=1e-8;

    MLP() = default;
    
    void setup(std::vector<int> npl_, std::vector<string> activ_)
    {
        this->npl = npl_;
        this->activ = activ_;

        this->W =           std::vector<matrix<double>>(npl.size());
        this->bias =        std::vector<matrix<double>>(npl.size());
        this->W_change =    std::vector<matrix<double>>(npl.size());
        this->delta =       std::vector<matrix<double>>(npl.size(), matrix<double>());
        this->S =           std::vector<matrix<double>>(npl.size(), matrix<double>());
        for (int i = 1; i < npl.size(); ++i)
        {
            int ii = npl[i-1], jj = npl[i];

            this->W[i] =          generate_matrix(ii, jj);
            this->W_change[i] =   generate_matrix(ii, jj);
            this->bias[i] =       matrix<double>(1, jj, 0);
        }

        for (const auto& w : W)
            cout << get_shape(w) << " ";
        cout << "\n";
        for (const auto& b : bias)
            cout << get_shape(b) << " ";
        cout << "\n";
        for (const auto& wc : W_change)
            cout << get_shape(wc) << " ";
        cout << "\n\n";
    }
    
    void split_data()
    {
        matrix<double> X = this->data["X"], Y = this->data["Y"];
        int rX = X.size1(), cX = X.size2(), rY = Y.size1(), cY = Y.size2();
        int index = 0.8*rX;

        this->data["X_train"] = subrange(X, 0, index, 0, cX);
        this->data["Y_train"] = subrange(Y, 0, index, 0, cY);
        
        this->data["X_test"] = subrange(X, index, int(0.9*rX), 0, cX);
        this->data["Y_test"] = subrange(Y, index, int(0.9*rX), 0, cY);

        this->data["X_validation"] = subrange(X, int(0.9*rX), rX, 0, cX);
        this->data["Y_validation"] = subrange(Y, int(0.9*rX), rX, 0, cY);
    }

    void f_act(matrix<double>& fv, int i)
    {
        activation_function[activ[i]](fv);
    }

    matrix<double> forward(matrix<double> XX)
    {
        matrix<double> output = XX;
        S[0] = output;
        delta[0] = output;
        for (int i = 1; i < npl.size(); ++i)
        {
            // cout << "\tlayer " << i << " -> " << get_shape(output) << " " << get_shape(W[i]) << "\n\t";
            output = prod(output, W[i]);
            // cout << get_shape(output) << endl;
            matrix_row<matrix<double>> bs(bias[i], 0);
            for (int j=0; j<output.size1(); j++)
            {   // sum bias in each row
                matrix_row<matrix<double>> mr(output, j);
                mr += bs;
            }
            // cout  << "neta:\n" << output << endl;
            f_act(output, i);
            // cout  << "f_act " << activ[i] << " bias:\n" << output << endl;
            S[i] = output;
            delta[i] = activ_derivatives[activ[i]](S[i]);    
            // cout  << "delta " << i << ": " << delta[i] << endl;
        }
        return output;
    }

    void backward(matrix<double> YY, matrix<double> YY_expected)
    {
        matrix<double> error = YY - YY_expected;
        int last_idx = delta.size() - 1;

        W_change[last_idx] = trans(prod(trans(delta[last_idx-1]), error));
        delta[last_idx] = error;

        for (int i = last_idx-1; i >= 1; i--)
        {
            auto Wt = trans(W[i+1]), St = trans(S[i+1]);
            delta[i] = element_prod(prod(delta[i+1], Wt), delta[i]);
            W_change[i] = prod(trans(delta[i]), S[i-1]);
        }

        for (int i = W.size()-1; i >= 1; i--)
        {
            auto wt = trans(W_change[i]);
            auto del = delta[i];
            matrix<double> colmeans(1, del.size2(), 0);
            for ( int k=0; k<del.size1(); k++ )
                for ( int j=0; j<del.size2(); j++ )     
                    colmeans(j) += del(k, j);

            W[i] -= learning_rate * wt;
            bias[i] = bias[i] - learning_rate * lambda * colmeans/del.size1();
        }
    }

    double cross_entropy(int j, const matrix<double>& Y, const matrix<double>& Y_expected)
    {
        double ce = 0;
        for (size_t i = 0; i < Y.size2(); i++)
                ce += std::isnan(-1* Y_expected(j, i) * log(Y(j, i))) ? 0 : -1* Y_expected(j, i) * log(Y(j, i));
        return ce;
    }
    double accuracy(matrix<double> Y, matrix<double> Y_expected)
    {
        double ce_prom = 0, max_el;
        for (size_t i = 0; i < Y.size1(); i++)
        {
            ce_prom += cross_entropy(i, Y, Y_expected);
        }
        return ce_prom / Y.size1();
    }


    double validation()
    {
        matrix<double> f = forward(data["X_validation"]);
        return accuracy(f, data["Y_validation"]);
    }

    void train()
    {
        int EPOCHS = 10000;
        int i = 0;
        while (true)
        {
            matrix<double> f = forward(data["X_train"]);
            backward(f, data["Y_train"]);
            auto train_err = accuracy(f, data["Y_train"]), val_err = validation();
            ERROR_STREAM << fixed << setprecision(3) << train_err << ", " << val_err << endl;
            if (i % 500 == 0)
                cout << fixed << setprecision(3) << train_err << ", " << val_err << endl;
            if (train_err < 0.05)
                    break;
            i++;
        }

        cout << fixed << setprecision(3) << forward(data["X_train"]) << endl << data["Y_train"] << endl;

        for (int i = 0; i < W.size(); ++i)
            export_(MODEL_PATH+"W_"+to_string(i)+".txt", W[i]);
        
        for (int i = 0; i < bias.size(); ++i)
            export_(MODEL_PATH+"bias_"+to_string(i)+".txt", bias[i]);
    }

    void conf_matrix(matrix<double> data)
    {
        
    }

    void test()
    {
        for (int i = 0; i < W.size(); ++i)
            import_(MODEL_PATH+"W_"+to_string(i)+".txt", W[i]);
        
        for (int i = 0; i < bias.size(); ++i)
            import_(MODEL_PATH+"bias_"+to_string(i)+".txt", bias[i]);
        
        matrix<double> train_confusion_matrix(10, 10, 0);
        export_(MODEL_PATH+"test_confusion_matrix.txt", train_confusion_matrix);
    }
};


void fit(MLP* mlp, int sample)
{
    mkdir(MODEL_PATH.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    
    string xx = DATA_PATH + "leedsbutterfly_6_stratified.csv", yy = DATA_PATH + "y_expected_stratified.csv";
    matrix<double> X = read_csv(xx, sample), Y = read_csv(yy, sample);
    
    std::vector<int> npl = {100, 90, 70, 50, 30, 20, 10};
    std::vector<string> fact = {"", "relu", "relu", "relu", "relu", "relu", "soft-max"};
    
    mlp->data["X"] = X;
    mlp->data["Y"] = Y;
    mlp->setup(npl, fact);
    mlp->split_data();
    // for (auto [k, v] : mlp->data)
    //     cout << k << " \t\t" << get_shape(v) << endl;
    mlp->train();
}

void test(MLP* mlp)
{
    mlp->test();
}

int main()
{
#ifndef TEST
    // freopen("input.txt", "r", stdin);
    // freopen("output.txt", "w", stdout);
#endif
    ios::sync_with_stdio(0);
    cin.tie(0);

    std::vector<int> samples = {100, 250, 500};

    for (const auto& s : samples)
    {
        MODEL_PATH = "./model_" + to_string(s) + "/";
        cout << MODEL_PATH << endl;
        auto mlp = new MLP();

        ERROR_STREAM = ofstream(MODEL_PATH + "error.csv", ios::out);
        fit(mlp, s);
        ERROR_STREAM.close();
    }
}