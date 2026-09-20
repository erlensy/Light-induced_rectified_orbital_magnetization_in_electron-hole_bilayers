/*
Adapted from:
C. P. Bridge, RIFeatures, src/struve.cpp
https://github.com/CPBridge/RIFeatures/blob/ca0a2d56e3c6d840b523572728bdc1bcf8284c4b/src/struve.cpp

RIFeatures states that the original Struve-function implementation
was written by J.-P. Moreau and was slightly modified for RIFeatures.

The underlying STVH0 routine originates from:
S. Zhang and J. Jin, Computation of Special Functions,
Wiley, 1996.
*/
#include <cmath>

double struveh0(double X)
{
	double A0,BY0,P0,PI,Q0,R,S,T,T2,TA0,SH0;
	int K, KM;
	bool negative_X = (X < 0.0);
	if(negative_X)
		X = -X;

	PI=3.141592653589793;
	S=1.0;
	R=1.0;
	if (X <= 20.0)
	{
		A0=2.0*X/PI;
		for (K=1; K<61; K++)
		{
			R=-R*X/(2.0*K+1.0)*X/(2.0*K+1.0);
			S=S+R;
			if (std::abs(R) < std::abs(S)*1.0e-12) break;
		}
		SH0=A0*S;
	}
	else
	{
		KM=int(0.5*(X+1.0));
		if (X >= 50.0)
			KM=25;
		for (K=1; K<=KM; K++)
		{
			R=-R*std::pow((2.0*K-1.0)/X,2);
			S=S+R;
			if (std::abs(R) < std::abs(S)*1.0e-12)
				break;
		}
		T=4.0/X;
		T2=T*T;
		P0=((((-.37043e-5*T2+.173565e-4)*T2-.487613e-4)*T2+.17343e-3)*T2-0.1753062e-2)*T2+.3989422793;
		Q0=T*(((((.32312e-5*T2-0.142078e-4)*T2+0.342468e-4)*T2-0.869791e-4)*T2+0.4564324e-3)*T2-0.0124669441);
		TA0=X-0.25*PI;
		BY0=2.0/std::sqrt(X)*(P0*std::sin(TA0)+Q0*std::cos(TA0));
		SH0=2.0/(PI*X)*S+BY0;
	}
	if(negative_X)
		SH0 = -SH0;
	return SH0;
}
