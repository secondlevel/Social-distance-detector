# include<iostream>
# include<vector>
using namespace std;

int main(void){

    Metrix m;
    cout << m.x(0) << endl;
    return 0;
}

class Metrix{

    private:  
        
        vector<int> image;

    public:

        ~Metrix(){ image.clear(); }

        float x(size_t it){
            return image.at(it);
        }


};