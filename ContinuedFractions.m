(* ::Package:: *)

(* ::Title:: *)
(*ContinuedFractions Package       *)


(* ::Subtitle:: *)
(*\[CapitalAAcute]lvaro Gonz\[AAcute]lez Hern\[AAcute]ndez - Universidad de Salamanca*)


(* ::Text:: *)
(*In this package I have coded some functions related to computations that involve continued fractions. These functions make use of some of Mathematica's intern functions such as ContinuedFraction, ContinuedFractionK, FromConvergents or PadeApproximants to perform calculations that are relevant in both the fields of complex analysis and number theory.*)
(**)


BeginPackage["ContinuedFractions`"];


(* ::Section:: *)
(*Usage and error messages*)


(* ::Subsection:: *)
(*Usage*)


(* ::Subsubsection:: *)
(*General algorithms for continued fractions*)


ForwardRecurrence::usage="ForwardRecurrence[n_, listA_, listB_] takes as parameters two lists of length at least n and n+1 and computes the n-th approximant via the forward recurrence method.";


EulerMinding::usage="EulerMinding[n_, listA_, listB_] takes as parameters two lists of length at least n and n+1 and computes the n-th approximant via the Euler-Minding method.";


BackwardRecurrence::usage="BackwardRecurrence[n_, listA_, listB_] takes as parameters two lists of length at least n and n+1 and computes the n-th approximant via the backward recurrence method.";


Approximants::usage="Approximants[n_, listA_, listB_] takes as parameters two lists of length at least n and n+1 and computes all the approximants until order n via the Euler Minding method.";


PartialND::usage="PartialND[n_, listA_, listB_] takes as parameters two lists of length at least n and n+1 and computes all the partial numerators and denominators until order n via the forward recurrence method.";


FromConvergents::usage="FromConvergents[n_,listP_,listQ_] takes as parameters two lists of length at least n+1 and computes the lists of coefficients of the continued fraction whose partial numerators and denominators are the arguments of the function. If the lists P and Q satisfy P[[i-1]]Q[[i]]-P[[i]]Q[[i-1]] for some i<n+2, it will return ComplexInfinity as part of the lists of coefficients.";


FromConvergentsK::usage="FromConvergentsK[P_,Q_] takes as parameters two pure functions and it returns two functions that describe the coefficients of the continued fraction whose partial numerators and denominators are P and Q.";


(* ::Subsubsection:: *)
(*Transformations of continued fractions*)


SumToCF::usage="SumToCF[F_] takes the pure function that determines the general term of a series and returns the expression of the coefficients of the continued fraction whose convergents are the partial sums of that series via Euler's identity.";


BauerMuir::usage="BauerMuir[n_,listA_,listB_,listG_] takes as parameters three lists of lengths n, n+1 and n+1 with the coefficients of a continued fraction and the coefficients for the Bauer-Muir transform and returns a list with the coefficients of the transformed continued fraction.";


BauerMuirK::usage=" BauerMuirK[A_,B_,G_] takes as parameters three pure functions that describe the coefficients of a continued fraction and the coefficients for the Bauer-Muir transform and returns a list with two pure functions that describe the coefficients of the transformed continued fraction.";


OnesDown::usage="OnesDown[n_,listA_,listB_] receives as parameters two lists of length n and n+1 with the coefficients of a continued fraction and returns the partial numerators of the equivalent continued fraction such that all the partial denominators are 1.";


OnesDownK::usage="OnesDownK[A_,B_] receives as parameters two pure functions that describe the coefficients of a continued fraction and returns a pure function with the partial numerators of the equivalent continued fraction such that all the partial denominators are 1.";


OnesUp::usage="OnesUp[n_,listA_,listB_] receives as parameters two lists of length n and n+1 with the coefficients of a continued fraction and returns the partial denominators of the equivalent continued fraction such that all the partial numerators are 1.";


OnesUpK::usage="OnesUpK[A_,B_] receives as parameters two pure functions that describe the coefficients of a continued fraction and returns a pure function with the partial denominators of the equivalent continued fraction such that all the partial numerators are 1.";


CanonicalEvenPart::usage="CanonicalEvenPart[n_,listA_,listB_] takes as parameters two lists of length at least 2n and 2n+1 that are the coefficients of a continued fraction and computes the lists of the first n coefficients of its canonical even part.";


CanonicalEvenPartK::usage="CanonicalEvenPartK[A_,B_] takes as parameters two pure functions that are the coefficients of a continued fraction and returns the coefficients of its canonical even part (as pure functions).";


CanonicalOddPart::usage="CanonicalOddPart[n_,listA_,listB_] takes as parameters two lists of length at least 2n+1 and 2n+2 that are the coefficients of a continued fraction and computes the lists of the first n coefficients of its canonical odd part.";


CanonicalOddPartK::usage="CanonicalOddPartK[A_,B_] takes as parameters two pure functions that are the coefficients of a continued fraction and returns the coefficients of its canonical odd part (as pure functions).";


Extend::usage="Extend[r_,p_,n_,listA_,listB_] takes two lists of length at least n and n+1 that are the coefficients of a continued fraction and returns the coefficients of the continued fraction whose approximants are the same until the position p, where the approximant is the element r, and after that the k+1-th approximant is the k-th of the original continued fraction.";


CFraction::usage="CFraction[n_,Fx_,x_] takes a natural number n and a expression defining a function Fx of x and returns a list with the n first terms of the regular C-fraction equivalent to Fx. It will likely fail if it is not given a normal function.";


(* ::Subsubsection:: *)
(*Convergence of continued fractions*)


ChordalD::usage="ChordalD[z1_,z2_] takes as parameters two complex numbers and returns its distance according to the chordal metric.";


SternStolz::usage="SternStolz[A_,B_] takes as parameters two pure functions that describe the coefficients of a continued fraction and returns the conditions under which the Stern-Stolz series of that continued fraction converges. True means the Stern-Stolz series converges so, the continued fraction diverges; whereas False means that the Stern-Stolz series diverge, and this may or may not imply the convergence of the continued fraction.";


(* ::Subsubsection:: *)
(*Continued fractions in number theory*)


EuclideanSCF::usage="EuclideanSCF[rat_] takes as argument a rational number (or a quotient between two Gaussian integers) and returns a list of the coefficients of the simple continued fraction expansion in the natural numbers (or in Gaussian Integers) through the Euclidean algorithm. The reason why this algorithm works for Gaussian Integers is that the functions Quotient and Mod are programmed to work with them.";


EuclideanSCFPolynomial::usage="EuclideanSCF[rat_,x_,mod_.] takes as argument a rational function over x (a quotient of two polynomials over x) and returns a list of the coefficients of the simple continued fraction expansion (polynomials) through the Euclidean algorithm. The third argument is an optional one that allows us to work with polynomials with coefficients in Z_n by adding the option: Modulus->n.";


SCFAlgorithm::usage="SCFAlgorithm[real_,n_] takes as arguments a real and a natural number and it returns the simple continued fraction expansion of that number until the n-th term. It also works for complex numbers as the function Floor is implemented for these as well (it returns the sum of the integer parts of the real and imaginary parts). It is worth noting that by changing this definition of Floor function, different simple continued fraction expansions for complex numbers can be obtained.";


CompleteQuotients::usage="CompleteQuotients[r_] takes a quadratic irrational as an argument and returns a finite list with all the possible complete quotients of r. The complete quotients that are found inside the inner brackets are those which repeat periodically. If it is given something different from a quadratic irrational, it will return a error message instead.";


LagrangeConstant::usage= "LagrangeConstant[r_] takes a quadratic irrational as an argument and returns its Lagrange constant. If it is given something different from a quadratic irrational, it will return a error message instead.";


PellsPossibilities::usage="PellsPossibilities[d_] takes a non-square integer and returns a list of all possible values of the form (-1)^n Q_n for which Pell's equation x^2-dy^2=(-1)^n Q_n has integer solutions. If it is given something different than an integer that is not a perfect square, it will return a error message instead.";


SmallestPellSolution1::usage="SmallerPellSolution1[d] takes a non-square integer and returns the list {x,y}, where x and y are the smallest positive solutions of x^2-dy^2=1.";


MarkovSolutions::usage="MarkovSolutions[n_] generates n triples of solutions to Markov's equation from the initial triple {1,2,5}.";


MarkovGraph::usage="MarkovGraph[n_,fontColor_,fontSize_,edgeColor_,ratio_] generates a graph with n triples of solutions to Markov's equation with the set colour of font for the numbers, that size of font, that color of edges and that aspect ratio";


(* ::Subsection:: *)
(*Error messages*)


WrongLength::error="The first list must have at least length `` and the second at least length ``";


WrongLength2::error="The first list must have at least length ``, the second at least length `` and the third one at least length ``";


WrongPosition::error="The chosen position: `` cannot be greater than the number of approximants: ``";


NotAQuadraticIrrational::error="The argument of this function must be a quadratic irrational";


NonSquareInteger::error="This function only works for integers that are not perfect squares";


MarkovGraphError::error="Please, choose at least 3 vertices!";


(* ::Section:: *)
(*General algorithms for continued fractions*)


Begin["`Private`"];


ForwardRecurrence[n_,listA_,listB_]:=
Module[{ A=Prepend[listA,1],B=listB,P={listB[[1]],listB[[2]]listB[[1]]+listA[[1]]},Q={1,listB[[2]]}},(*Initialization of variables*)
If[Length[A]<(n+1)||Length[B]<(n+1),Message[WrongLength::error,n,n+1];Return[]];(*Error message*)
For[i=3,i<=n+1,i++,AppendTo[P,B[[i]]P[[i-1]]+A[[i]]P[[i-2]]]; AppendTo[Q,B[[i]]Q[[i-1]]+A[[i]]Q[[i-2]]]];(*Sequential updating of the lists of partial numerators and denominators*)
Last[P/Q](*Returns n-th approximant*)]


EulerMinding[n_,listA_,listB_]:=
Module[{A=Prepend[listA,1],B=listB,Q={1,listB[[2]]},W={listB[[1]]},det=1},(*Initialization of variables*)
If[Length[A]<(n+1)||Length[B]<(n+1),Message[WrongLength::error,n,n+1];Return[]];(*Error message*)
For[i=3,i<=n+1,i++,AppendTo[Q,B[[i]]Q[[i-1]]+A[[i]]Q[[i-2]]]];(*We first create a list of the partial denominators*)
For[i=2,i<=n+1,i++,det=det*(-A[[i]]); AppendTo[W,W[[i-1]]-(det)/(Q[[i]]Q[[i-1]])]];(*From that, we compute the approximants using Euler Minding's formula*)
Last[W]]


BackwardRecurrence[n_,listA_,listB_]:=
Module[{A=Prepend[listA,1],B=listB,L={0}},
If[Length[A]<(n+1)||Length[B]<(n+1),Message[WrongLength::error,n,n+1];Return[]];(*Error message*)
For[i=n+1,i>1,i--,L=A[[i]]/(B[[i]]+L)]; (*We create a list backwards by applying sequentially the iteration L*)
First[L+B[[1]]]]


Approximants[n_,listA_,listB_]:=
Module[{A=Prepend[listA,1],B=listB,Q={1,listB[[2]]},W={listB[[1]]},det=1}, (*This function mimics the behaviour of EulerMinding, but instead of returning the last approximant, it returns a list of all approximants*)
If[Length[A]<(n+1)||Length[B]<(n+1),Message[WrongLength::error,n,n+1];Return[]];(*Error message*)
For[i=3,i<=n+1,i++,AppendTo[Q,B[[i]]Q[[i-1]]+A[[i]]Q[[i-2]]]];
For[i=2,i<=n+1,i++,det=det*(-A[[i]]); AppendTo[W,W[[i-1]]-(det)/(Q[[i]]Q[[i-1]])]];W]


PartialND[n_,listA_,listB_]:=
Module[{ A=Prepend[listA,1],B=listB,P={listB[[1]],listB[[2]]listB[[1]]+listA[[1]]},Q={1,listB[[2]]}},(*This function mimics the behaviour of ForwardRecurrence, but instead of returning the last approximant, it returns two lists: one of the partial numerators and another of the partial denominators*)
If[Length[A]<(n+1)||Length[B]<(n+1),Message[WrongLength::error,n+1,n+1];Return[]];(*Error message*)
For[i=3,i<=n+1,i++,AppendTo[P,B[[i]]P[[i-1]]+A[[i]]P[[i-2]]];AppendTo[Q,B[[i]]Q[[i-1]]+A[[i]]Q[[i-2]]]];
{P,Q}]


FromConvergents[n_,listP_,listQ_]:=
Module[{P=listP, Q=listQ,A={-listP[[1]] listQ[[2]]+listP[[2]]listQ[[1]]},B={listP[[1]],listQ[[2]]}},(*We initialize the variables*) 
If[Length[P]<(n+1)||Length[Q]<(n+1),Message[WrongLength::error,n+1,n+1];Return[]];(*Error message*)
A=Join[A,Table[-(P[[i]]Q[[i+1]]-P[[i+1]]Q[[i]])/(P[[i-1]]Q[[i]]-P[[i]]Q[[i-1]]),{i,2,n}]]; (*We apply the formulas*)
B=Join[B,Table[(P[[i-1]]Q[[i+1]]-P[[i+1]]Q[[i-1]])/(P[[i-1]]Q[[i]]-P[[i]]Q[[i-1]]),{i,2,n}]];
 {A,B} (*We return lists A and B*)] 


FromConvergentsK[P_,Q_]:={Function[n,If[n==1,-P[0] Q[1]+P[1]Q[0],-(P[n-1]Q[n]-P[n]Q[n-1])/(P[n-2]Q[n-1]-P[n-1]Q[n-2])]],
Function[n,If[n==1,Q[1],(P[n-2]Q[n]-P[n]Q[n-2])/(P[n-2]Q[n-1]-P[n-1]Q[n-2])]]}(*The definition of pure functions inside a module is done through the Function command*)


(* ::Section:: *)
(*Transformations for continued fractions*)


SumToCF[F_]:={Function[n,If[n==1,F[1],-F[n]/F[n-1]]],Function[n,If[n==1,1,1+F[n]/F[n-1]]]}


BauerMuir[n_,listA_,listB_,listG_]:=
Module[{A=listA,B=listB,G=listG,L,C={listA[[1]]-listG[[1]](listB[[2]]+listG[[2]])},D={listB[[1]]+listG[[1]],listB[[2]]listG[[2]]}},(*We set the first and the first two elements of C and D respectively*)
If[Length[A]<n||Length[B]<(n+1)||Length[G]<(n+1),Message[WrongLength2::error,n,n+1,n+1];Return[]];(*Error message*)
L=Table[A[[i]]-G[[i]](B[[i+1]]+G[[i+1]]),{i,1,n}]; (*Creates a list with the lambdas*)
C=Join[C,Table[A[[i-1]]*L[[i]]/L[[i-1]],{i,2,n}]];
D=Join[D,Table[B[[i+1]]+G[[i+1]]-G[[i-1]]*L[[i]]/L[[i-1]],{i,2,n}]];(*Applies the definition of th Bauer-Muir transform*)
{C,D}]


BauerMuirK[A_,B_,G_]:={Function[n,If[n==1,A[1]-G[0](B[1]+G[1]),A[n-1]*(A[n]-G[n-1](B[n]+G[n]))/(A[n-1]-G[n-2](B[n-1]+G[n-1]))]], Function[n,If[n<2,B[n]+G[n] ,B[n]+G[n]-G[n-2]*(A[n]-G[n-1](B[n]+G[n]))/(A[n-1]-G[n-2](B[n-1]+G[n-1]))]]}(*It has the same implementation as the version for lists ith the exception of the indices of the functions that change due to the fact that in Mathematica the lists begin with 1 and our coefficients begin with 0*)


OnesDown[n_,listA_,listB_]:=
Module[{A=listA, B=listB,C={listA[[1]]/listB[[2]]}},
If[Length[A]<n||Length[B]<(n+1),Message[WrongLength::error,n,n+1];Return[]];(*Error message*)
C=Join[C,Table[A[[i]]/(B [[i]]B[[i+1]]),{i,2,n}]]; (*This is the transform*)
C]


OnesDownK[A_,B_]:=Function[n,If[n==1,A[n]/B[n],A[n]/(B [n-1]B[n])]]


OnesUp[n_,listA_,listB_]:=
Module[{A=listA, B=listB, Z={1,1/listA[[1]]},D={listB[[1]]}},
If[Length[A]<n||Length[B]<(n+1),Message[WrongLength::error,n,n+1];Return[]];(*Error message*)
Z=Join[Z,Table[If[EvenQ[i],Product[A[[2j-1]]/A[[2j]],{j,1,i/2}],Product[A[[2j]]/A[[2j+1]],{j,1,(i-1)/2}]/A[[1]]],{i,2,n}]];
D=Join[D,Table[Z[[i+1]]B[[i+1]],{i,1,n}]];(*To define this, we treat differently the case where the index is odd from when it is even*)
D]


OnesUpK[A_,B_]:=Function[n,B[n]Product[A[k]^(-1)^(n-k+1),{k,1,n}]]


CanonicalEvenPart[n_,listA_,listB_]:=
Module[{A=listA,B=listB,C={listA[[1]]listB[[3]]},D={listB[[1]],listA[[2]]+listB[[2]]listB[[3]]}},(*We initialize the lists C and D*)
If[Length[A]<2n||Length[B]<(2n+1),Message[WrongLength::error,2n,2n+1];Return[]];(*Error message*)
C=Join[C,Table[-A[[2i-2]]A[[2i-1]]B[[2i+1]]/B[[2i-1]],{i,2,n}]];D=Join[D,Table[A[[2i]]+B[[2i]]B[[2i+1]]+A[[2i-1]]B[[2i+1]]/B[[2i-1]],{i,2,n}]];{C,D}]


CanonicalEvenPartK[A_,B_]:={Function[n,If[n==1,A[1]B[2],(A[2n-2]A[2n-1]B[2n])/B[2n-2]]],
Function[n,If[n==1,A[2]+B[1]B[2],A[2n]+B[2n-1]B[2n]+(A[2n-1]B[2n])/B[2n-2]]]}


CanonicalOddPart[n_,listA_,listB_]:=(*You need lists of length 2n+1 and 2n+2*)
Module[{A=listA,B=listB,C={-listA[[1]]listA[[2]]listB[[4]]/listB[[2]],-listA[[3]] listA[[4]]listB[[6]]listB[[2]]/listB[[4]]},D={listB[[1]]+listA[[1]]/listB[[2]],listA[[3]]listB[[2]]+listB[[2]]listB[[3]]listB[[4]]+listA[[2]]listB[[4]]}},(*We initialize the lists C and D*)
If[Length[A]<(2n+1)||Length[B]<(2n+2),Message[WrongLength::error,2n+1,2n+2];Return[]];(*Error message*)
C=Join[C,Table[-A[[2i-1]]A[[2i]]B[[2i+2]]/B[[2i]],{i,3,n}]];D=Join[D,Table[A[[2i+1]]+B[[2i+1]]B[[2i+2]]+A[[2i]]B[[2i+2]]/B[[2i]],{i,2,n}]];{C,D}]


CanonicalOddPartK[A_,B_]:={Function[n,If[n==2,-A[3]A[4]B[1]B[5]/B[3],-A[2n-1]A[2n]B[2n+1]/B[2n-1]]],
Function[n,If[n==0,B[0]+A[1]/B[1],If[n==1,A[3]B[1]+B[1]B[2]B[3]+A[2]B[3],A[2n+1]+B[2n]B[2n+1]+A[2n]B[2n+1]/B[2n-1]]]]}


Extend[r_, p_,n_,listA_,listB_]:=Module[{A=listA,B=listB, ND=PartialND[p,listA,listB],rho,C,D},
If[Length[A]<n||Length[B]<(n+1),Message[WrongLength::error,n,n+1];Return[]];(*First error message*)
If[p>n,Message[WrongPosition::error,p,n];Return[]]; (*Second error message*)
rho=(ND[[1,p+1]]-ND[[2,p+1]]*r)/(ND[[1,p]]-ND[[2,p]]*r);
C=Table[Which[i<=p,A[[i]],i==p+1,rho,i==p+2,-A[[p+1]]/rho,i>=p+3,A[[i-1]]],{i,1,n+1}];
D=Table[Which[i<=p,B[[i]],i==p+1,B[[p+1]]-rho,i==p+2,1,i==p+3,B[[p+2]]+A[[p+1]]/rho,i>=p+4,B[[i-1]]],{i,1,n+2}];
{C,D}]


CFraction[n_,Fx_,x_]:=Module[{k,pade1,pade2,pade,cf,a0=Fx/.{x->0}},
pade1=Table[PadeApproximant[Fx,{x,0,{k,k}}],{k,0,Ceiling[(n+1)/2]}]; (*Computes the terms in the diagonal of the Pad\[EAcute] table*)
If[a0==0,pade1[[1]]=0]; (*The function PadeApproximant fails to return the approximant (0,0) if this is zero, so we have implemented this case separatedly*)
pade2=Table[PadeApproximant[Fx,{x,0,{k+1,k}}],{k,0,Floor[(n+1)/2]}]; (*Computes the terms under the diagonal*)
pade=Table[If[OddQ[k],pade1[[(k+1)/2]],pade2[[k/2]]],{k,1,n+1}]; (*Join both to form the staircase sequence*)
cf=Simplify[FromConvergents[n,Numerator[pade],Denominator[pade]]];
Prepend[OnesDown[n,cf[[1]],cf[[2]]],a0] (*Uses FromConvergents to get the continued fraction and OnesDown to get the partial denominators equal 1*)
]


(* ::Section:: *)
(*Convergence of continued fractions*)


ChordalD[z1_,z2_]:=Which[z1==Infinity &&z2==Infinity,0,z1==Infinity,2/Sqrt[1+Norm[z2]^2],z2==Infinity,2/Sqrt[1+Norm[z1]^2],True,2Norm[z1-z2]/(Sqrt[1+Norm[z1]^2]*Sqrt[1+Norm[z2]^2])]


SternStolz[A_,B_]:=Simplify[SumConvergence[Abs[B[2 n]Product[A[2k-1]/A[2k],{k,1,n}]],n]&&SumConvergence[Abs[B[2 n+1]/A[1]Product[A[2k]/A[2k+1],{k,1,n}]],n]](*For this function, we divide into even and odd parts to make the study of the convergence of the series easier for Mathematica.*)


(* ::Section:: *)
(*Continued fractions in number theory*)


EuclideanSCF[rat_]:=Module[{num=Numerator[Simplify[rat]],den=Denominator[Simplify[rat]],k,Q={}},(*The following piece of code implements the Euclidean algorithm.The variable k is a dummy variable used to store and upgrade the values of num and den after every iteration*)
While[den!=0,AppendTo[Q,Quotient[num,den]];k=Mod[num,den];num=den;den=k];
Q (*This is the list with the coefficients*)]


EuclideanSCFPolynomial[rat_,x_]:=Module[{num=Numerator[Simplify[rat]],den=Denominator[Simplify[rat]],r,Q={}},
While[Exponent[den,x]!=-Infinity,AppendTo[Q,PolynomialQuotient[num,den,x]];(*As polynomial=0 does not have a logical value, if we used den\[NotEqual]0 the code would not work, that is why we use Exponent[den,x]\[NotEqual]-Infinity, as 0 is the only polynomial with degree -Infinity (according to Mathematica)*)
r=PolynomialRemainder[num,den,x];num=den;den=r];
Q (*The rest of the implementation is the same as EuclideanSCF, with the only difference that PolynomialQuotient and PolynomialReminder are used instead of Quotient and Mod*)]

EuclideanSCFPolynomial[rat_,x_,mod_]:=Module[{num=Numerator[Simplify[rat]],den=Denominator[Simplify[rat]],r,Q={}},
While[Exponent[den,x]!=-Infinity,AppendTo[Q,PolynomialQuotient[num,den,x,mod]];
r=PolynomialRemainder[num,den,x,mod];num=den;den=r]; (*This includes the option to work in finite fields*)
Q]


SCFAlgorithm[real_,n_]:=Module[{B={},r=real},(*This piece of code computes the complete quotients r and stores their integer parts in the list B*)
For[i=0,i<=n,i++,B=AppendTo[B,Floor[r]];r=1/(r-B[[i+1]])]; B]


CompleteQuotients[r_]:=Module[{q,l,period,lperiod,list={},x,rep},
If[!QuadraticIrrationalQ[r],Message[NotAQuadraticIrrational::error];Return[]]; (*Error message*)
q=ContinuedFraction[r]; (*Stores the continued fraction expansion of r*)
period=Last[q]; (*Stores the periodic part of that expansion*)
l=Length[q]-1;
lperiod=Length[period];x=q;
For[i=1,i<=l-1,i++,AppendTo[list,Delete[x,1]];x=Last[list]]; (*This loop produces a list where the elements are the result of removing the first digits of the continued fraction expansion of r (this corresponds to the definition of complete quotients)*)
rep=Table[{RotateLeft[period,n]},{n,0,lperiod-1}]; (*This produces a list in which we find all the possible cyclic permutations of the periodic part of the expansion. This corresponds to the expansion of all the complete quotients that repeat periodically*)
list=FromContinuedFraction/@list; 
rep=FromContinuedFraction/@rep; (*These past two lines generate the real numbers associated to the lists that we have created*)
Join[list,{rep}] (*The union of these lists (the second one inside brackets) give us our desired list as a result*)
]


LagrangeConstant[r_]:=Module[{q,period,rep,lperiod,repinv}, (*To compute this, we will use the formula that calculates it from the continued fraction expansion*)
If[!QuadraticIrrationalQ[r],Message[NotAQuadraticIrrational::error];Return[]]; (*Error message*)
q=ContinuedFraction[Abs[r]];(*Stores the continued fraction expansion of the absolute value of r. The absolute value is necessary for the function to be able to deal with negative numbers*)
period=Last[q]; (*Stores the periodic part of that expansion*)
lperiod=Length[period];
rep=Table[{RotateLeft[period,n]},{n,0,lperiod-1}]; (*This produces a list of the expansions of all complete quotients that repeat periodically [b_n,b_n+1,...]*)
repinv=Reverse[rep,3]; 
repinv=Prepend[0]/@repinv ; (*This reverses the previous list and adds a zero before it, so we get the possible limits when n\[Rule]Infinity of [0,b_n,...,b_1,b_0]*)
rep=FromContinuedFraction/@rep;
repinv=FromContinuedFraction/@repinv;(*We transform the continued fractions into real numbers*)
Max[Simplify[rep+repinv]] (*The Lagrange constant is the greatest element in the list that results when we add the elements of the lists rep and repinv*)
]


PellsPossibilities[d_]:=Module[{sq=Sqrt[d],period,lperiod,rep,rep2},
If[IntegerQ[Sqrt[d]],Message[NonSquareInteger::error];Return[]]; (*Error message*)
period=Last[ContinuedFraction[sq]];
lperiod=Length[period];
rep=Table[{RotateLeft[period,i]},{i,0,lperiod-1}]; (*This produces a list of the expansions of all complete quotients of sqrt(d) that repeat periodically *)
rep=FromContinuedFraction/@rep; (*We transform the continued fractions into real numbers*)
rep2=Join[rep,rep]; 
Sort[DeleteDuplicates[Table[(-1)^k*Denominator[rep2[[k]]],{k,1,2lperiod}]]](*As some continued fraction expansions have a periodic part of odd length, to compute all possibles (-1)^n Q_n, we need to iterate twice the length of the period. After that, we return the solutions sorted and without duplicates*)]


SmallestPellSolution1[d_]:=Module[{rd,q,period,k=0,n=0,u},
If[IntegerQ[Sqrt[d]],Message[NonSquareInteger::error];Return[]]; (*Error message*)
q= ContinuedFraction[Sqrt[d]];
If[Floor[Sqrt[d]]^2-d==1,u={Floor[Sqrt[d]],1},period=Last[q]; (*If the 0-th approximant is the solution, it stops*)
While[k!=1,k=(-1)^(n+1) Denominator[FromContinuedFraction[{period}]];period=RotateLeft[period,1];n=n+1;]; 
u=NumeratorDenominator[FromContinuedFraction[ContinuedFraction[Sqrt[d],n]]]](*If not, it rotates the period until the (-1)^(n+1)Q_(n+1)=1 and it returns the n-th partial numerator and n-th partial denominators which are the smallest solutions*)]


MarkovSolutions[n_]:=Module[{list={{1,2,5}}},For[i=2,i<=n,i++,AppendTo[list,If[EvenQ[i],{list[[Floor[i/2],1]],list[[Floor[i/2],3]],3*list[[Floor[i/2],1]]*list[[Floor[i/2],3]]-list[[Floor[i/2],2]]},{list[[Floor[i/2],2]],list[[Floor[i/2],3]],3*list[[Floor[i/2],2]]*list[[Floor[i/2],3]]-list[[Floor[i/2],1]]}]]];list]


WhitePanel[label_]:=Panel[label,Background->White,FrameMargins->-2] (*This is an auxiliary function to add a white panel under the numbers of the graph so it looks better*)

MarkovGraph[n_,fontColor_,fontSize_,edgeColor_,ratio_]:=Module[{M=MarkovSolutions[n-2],G},(*We write MarkovSolutions[n-2], because the two solutions {1,1,1},{1,1,2} will be added later*)
     If[n<3,Message[MarkovGraphError::error];Return[]];
     G=Graph[Join[{{1,1,1}\[UndirectedEdge]{1,1,2},{1,1,2}\[UndirectedEdge]{1,2,5}},Table[M[[i]]\[UndirectedEdge]M[[2i]],{i,1,Floor[(n-2)/2]}],Table[M[[i]]\[UndirectedEdge]M[[2i+1]],{i,1,Floor[(n-3)/2]}]],(*We create the graph by joining the triples*)
VertexLabels->{"Name",Placed[Automatic,Center,WhitePanel]},VertexSize->Tiny,GraphLayout->{"LayeredEmbedding","Orientation"->Left},VertexLabelStyle->Directive[fontColor,fontSize],VertexStyle->White, EdgeStyle->edgeColor, AspectRatio->ratio](*We set up the vertices labels and the rest of stylistic options*)]


End[];


EndPackage[];
