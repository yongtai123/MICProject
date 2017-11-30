function Adjacency = getAdjacency()
  size = 6;
  Adjacency = zeros(size);
  Adjacency(1,4) = 1;
  Adjacency(4,1) = 1;
  Adjacency(4,2) = 1;
  Adjacency(2,4) = 1;
  Adjacency(1,5) = 1;
  Adjacency(5,1) = 1;
  Adjacency(2,5) = 1;
  Adjacency(5,2) = 1;
  Adjacency(3,6) = 1;
  Adjacency(6,3) = 1;
  Adjacency(3,5) = 1;
  Adjacency(5,3) = 1;
  Adjacency(2,1) = 1;
  Adjacency(1,2) = 1;
  Adjacency(4,6) = 1;
  Adjacency(6,4) = 1;
  Adjacency(2,3) = 1;
  Adjacency(3,2) = 1;
  Adjacency(3,4) = 1;
  Adjacency(4,3) = 1;
  Adjacency(4,5) = 1;
  Adjacency(5,4) = 1;
  Adjacency(5,6) = 1;
  Adjacency(6,5) = 1;
  Adjacency(6,1) = 1;
  Adjacency(1,6) = 1;
  Adjacency(1,3) = 1;
  Adjacency(3,1) = 1;
