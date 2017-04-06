#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <iomanip>
#include <stdlib.h>   
#include <cstdlib>
#include <cstring>
#include <cmath>
using namespace std;
// input&bias 10 att + output 1 att
//bias => input_att1

void	randomWeight();
void    readData();

double	hiddenNode(double input_data[10][864],double w_input[3][10],int node ,int _Record);
double	outputNode(double output_hidden[4],double w_hidden[4]);
bool isRedundance(int frames[],int data,int frames_count);
void Switchdata();
void partition();

double  FindOutputGradian(double errror , double y );
double  FindHiddenGradian(double gradianOutput , double y ,double w);

void ReWeightHidden();
void ReWeightInput(int row);


int input_Record = 958;
int input_att_count = 10;
int hiddenNode_count = 4;

double input_data[10][958];
double Newinput_data[10][958];
double output_hidden[4];
double in[958];
double w_input[3][10];
double w_hidden[4];
double output[958];

double Learning_Rate = -0.9;
double d = 1;
double error[958];

double out_gradian;
double hidd_gradian[4];

double trainData[10][864];
double testData[10][94];






int main(){

    srand( (unsigned)time( NULL ));
	readData();// read cvs file 	
	randomWeight();
    int numberofcycle=0;

   
   for(int t = 0 ; t< 10 ;t++)
   {
        cout <<"cycly : "<< t+1 <<endl;
        Switchdata();
        partition();

        //start process 
        for(int c = 0 ;c<10 ;c++)
        {
            cout << "hop" << c+1 <<endl;
                for(int j = 0; j< 864 ; j++ ) 
            {
                output_hidden[0] = 1;
                for(int i=1; i< hiddenNode_count; i++)
                {
                    output_hidden[i] = hiddenNode(trainData,w_input,i,j);

                }

                output[j] = outputNode(output_hidden,w_hidden); // output of 1 record
                error[j] = d - output[j];
                out_gradian = FindOutputGradian(error[j],output[j]);
                
                for(int k = 1; k<4 ;k++)
                {
                    hidd_gradian[k] = FindHiddenGradian(out_gradian,output_hidden[k],w_hidden[k]);
                }

                ReWeightHidden();
                ReWeightInput(j);


                cout << j+1 << " : " << output[j]<<endl;
                
            }
            numberofcycle++;

        }  

   }
    cout << "number : " << numberofcycle << endl;

    
    //end process

    
    

    //print weight
    // for(int i = 0 ; i < 3 ;i++)
    // {
    //     for(int j = 0  ;j<10 ; j++)
    //     {
    //         cout << w_input[i][j] << " " ;
    //     }
    //     cout << endl;

    // }
    // cout << "------------"<<endl;

    // for(int j = 0  ;j<10 ; j++)
    //     {
    //         cout << w_hidden[j] <<endl;
    //     }
	
   

}

void randomWeight(){

    for (int i = 0; i < hiddenNode_count; i++) // random input weight
	{
		for(int j =0; j<10;j++)
		{
			w_input[i][j] = (double)rand()/RAND_MAX;
		}
		
    }

    for(int i = 0 ; i < hiddenNode_count ;i++)//random hidden weight
    {
    	w_hidden[i] = (double)rand()/RAND_MAX;
    }
}

void readData(){
	string txt;
    int x=0;
    int i=0;
    ifstream file("dataset.csv");
    while(getline(file,txt)){
        
        
        if(txt[i]!=',')
        {                

           input_data[1][x] = double(txt[0]-48);
            input_data[2][x] = double(txt[2]-48);
            input_data[3][x] = double(txt[4]-48);
            input_data[4][x] = double(txt[6]-48);
            input_data[5][x] = double(txt[8]-48);
            input_data[6][x] = double(txt[10]-48);
            input_data[7][x] = double(txt[12]-48);
            input_data[8][x] = double(txt[14]-48);
            input_data[9][x] = double(txt[16]-48);
            input_data[10][x] = double(txt[18]-48);
            in[x] = double(txt[20]-48);

        }
        else 
        {
            i++;
        }     
        if(i== txt.length())
        {        
            i=0;
        }       
        x++;


    }



    // for(int i=0;i<958;i++)
    // {
    //     cout << i+1 << " : "<<" "<< att1[i] << " "<<att2[i]<<" " << att3[i]
    //     //<<" att4= "<< att4[i] << " att5 = "<<att5[i]<<" att6 = " << att6[i]
    //     //<<" att7= "<< att7[i] << " att8 = "<<att8[i]<<" att9 = " << att9[i]
    //    // <<"att10= "<< att10[i] 
    //     << endl;

    // }

    // for (int i = 0; i <958;i++)
    // {
    // 	cout <<i +1<<" : "<< data[2][i]<< endl;;
    // }
}

double hiddenNode(double input_data[10][864],double w_input[3][10],int node ,int _Record){
	double v =0.0;
	const double e =  2.71828;

	for (int i = 0; i < input_att_count; i++)
	{
		v += input_data[i][_Record] * w_input[node-1][i];
	}


	return (1/(1+pow(e,-v)));
}
double	outputNode(double output_hidden[4],double w_hidden[4]){
	double v =0.0;
	const double e =  2.71828;

	for (int i = 0; i < hiddenNode_count; i++)
	{
		v += output_hidden[i] * w_hidden[i];
	}


	return (1/(1+pow(e,-v)));
}
double FindOutputGradian(double errror , double y ){
    return errror * y * (1-y);

}

double FindHiddenGradian(double gradianOutput , double y ,double w){
    return y*(1-y)*gradianOutput*w;

}
void ReWeightHidden()
{
    for(int i = 0; i < 4 ; i++)
    {
        w_hidden[i] += ((-Learning_Rate) * out_gradian * output_hidden[i]);

    }

}

void ReWeightInput(int row)
{
    //w_input[3][10];

    for(int i = 0 ; i < 3 ; i++)
    {
        for(int j = 0 ; j < 10 ; j++)
        {
            w_input[i][j] += ((-Learning_Rate) * hidd_gradian[i] *  input_data[j][row]);
        }
    }
    
}


bool isRedundance(int frames[],int data,int frames_count){

    bool check = false;
    for(int i=0; i<frames_count ;i++)
    {
        if(data == frames[i])
        {
            check = true;
        }
    }
    return check;
}

void Switchdata()
{

    //cout << "--------------------" << endl;
    int randarray[958];
    int zero =0;

    for(int i = 0 ; i < 958 ;)
    {
        int test = rand()%958;
        if(test == 0 )
        {
            zero++;
        }

        
        if(isRedundance(randarray,test,958)==false || zero == 1)
        {
            randarray[i] = test;
            //cout <<  randarray[i]<<endl;
            i++;

        }
        if(test == 0 )
        {
            zero++;
        }
        
    }
    

    int row =10;
    int col = 958;

    for(int i =0 ; i< row ;i++)
    {
        for(int j = 0 ; j<col;j++)
        {
            Newinput_data[i][j] = input_data[i][randarray[j]];
            //cout << "array " << arrold[i][j] << "arraynew " << arrnew[i][j]<<endl;
        }
    }


}
void partition()
{
    int col = 0;

    for(int row =0; row< 10 ; row++)
    {
        for(; col< 864 ; col++)
        {  
             trainData[row][col] = Newinput_data[row][col]; 

        }

    }
    
    for(int row = 0; row<10 ; row++)
    {
        for(; col<958 ; col++)
        {
            testData[row][col] = Newinput_data[row][col];
        }
    }

}















