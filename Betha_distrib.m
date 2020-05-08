X = 0:.01:1;
y1 = betapdf(X,2,4);
plot(X,y1)
xlabel('PDF of a beta distribution with a=2 and b=4') 