require 'nn'
require 'rnn'
--require 'ReverseTable'

local function loadhash(filename, maxidx)
   print(string.format('loading <%s>', filename))
   local hash = {}
   local idx = 0
   for key in io.lines(filename) do
      idx = idx + 1
      if maxidx and maxidx > 0 and idx > maxidx then
	 break
      end
      table.insert(hash, key)
      hash[key] = idx
   end
   return hash
end

function checklm(words, hash, params)
   local w = 'expression'
   
   local cat = words[hash[w]]
   
   local dists = {}
   if not params.lmsum then
      for i=1,params.nword do
	 table.insert(dists, {i, words[i]:dist(cat)})
      end
      table.sort(dists, function(a, b)
		    return a[2] < b[2]
      end)
   else
      local cosdis = nn.CosineDistance()
      for i=1,params.nword do
	 table.insert(dists, {i, cosdis:forward({words[i],cat})[1]})
      end
      table.sort(dists, function(a, b)
		    return a[2] > b[2]
      end)
   end
   print('[lm] check words closest from <' .. w .. '>' )
   for i=1,10 do
      print(string.format('[lm]  -- %s (%g)', hash[dists[i][1]], dists[i][2]))
   end
end

function get_par(params, lkts, dropout, fixe)
   local par = nn.ParallelTable()
   if params.dropout~=0 and (params.dp==1 or params.dp==3) then
      if fixe then
	 local lkt = lkts.words:clone()
	 local oldaccgradparameters = lkt.backwardUpdate
	 function lkt.updateParameters(self, input, gradOutput, scale)
	 end
	 function lkt.accUpdateGradParameters(self, input, gradOutput, scale)
	 end
	 function lkt.accGradParameters(self, input, gradOutput, scale)
	 end
	 local d = nn.Dropout(params.dropout); table.insert(dropout, d); par:add(nn.Sequential():add(lkt):add(d))
      else
	 local d = nn.Dropout(params.dropout); table.insert(dropout, d); par:add(nn.Sequential():add(lkts.words:clone('weight','bias')):add(d))
      end
      if params.tfsz~=0 then local d = nn.Dropout(params.dropout); table.insert(dropout, d); par:add(nn.Sequential():add(lkts.entitytags:clone('weight','bias')):add(d)) end
      if params.pfsz~=0 then local d = nn.Dropout(params.dropout); table.insert(dropout, d); par:add(nn.Sequential():add(lkts.pos:clone('weight','bias')):add(d)) end
      if params.rdfsz~=0 then
	 local d = nn.Dropout(params.dropout); table.insert(dropout, d); par:add(nn.Sequential():add(lkts.relativedistance1:clone('weight','bias')):add(d))
	 local d = nn.Dropout(params.dropout); table.insert(dropout, d); par:add(nn.Sequential():add(lkts.relativedistance2:clone('weight','bias')):add(d))
      end
      if params.dtfsz~=0 and params.dt==1 then local d = nn.Dropout(params.dropout); table.insert(dropout, d); par:add(nn.Sequential():add(lkts.deptypes:clone('weight','bias')):add(d)) end
   else
      if fixe then
	 local lkt = lkts.words:clone()
	 local oldaccgradparameters = lkt.backwardUpdate
	 function lkt.updateParameters(self, input, gradOutput, scale)
	 end
	 function lkt.accUpdateGradParameters(self, input, gradOutput, scale)
	    --print("hup")
	 end
	 function lkt.accGradParameters(self, input, gradOutput, scale)
	    --print("hop")
	 end
	 par:add(lkt)
      else
	 par:add(lkts.words:clone('weight','bias'))
      end
      if params.tfsz~=0 then par:add(lkts.entitytags:clone('weight','bias')) end
      if params.pfsz~=0 then par:add(lkts.pos:clone('weight','bias')) end
      if params.rdfsz~=0 then par:add(lkts.relativedistance1:clone('weight','bias')):add(lkts.relativedistance2:clone('weight','bias')) end
      if params.dtfsz~=0 and params.dt==1 then par:add(lkts.deptypes:clone('weight','bias')) end
   end
   par:add(lkts.entities:clone('weight','bias'))
   
   return par
end
   

function createnetworks(params, datas)
   local lkts = {}
   local words = nn.LookupTable(params.nword, params.wfsz)
   lkts.words = words
   local entities = nn.LookupTable(5, params.efsz)
   lkts.entities = entities
   --1:Padding 2:Other 3:entity1 4:entity2 5:node
   local entitytags, deptypes, relativedistance1, relativedistance2
   if params.tfsz~=0 then
      entitytags = nn.LookupTable(#datas[1].entityhash, params.tfsz)
      lkts.entitytags = entitytags
   end
   if params.pfsz~=0 then
      pos = nn.LookupTable(#datas[1].poshash, params.pfsz)
      lkts.pos = pos
   end
   if params.rdfsz~=0 then
      relativedistance1 = nn.LookupTable(300, params.rdfsz)
      relativedistance2 = nn.LookupTable(300, params.rdfsz)
      lkts.relativedistance1 = relativedistance1
      lkts.relativedistance2 = relativedistance2
   end
   if params.dtfsz~=0 then
      deptypes = nn.LookupTable(#datas[1].deptypehash, params.dtfsz)
      lkts.deptypes = deptypes
   end
   
   if params.lm then
      print('loading lm')
      local wordhash = datas[1].wordhash
      
      local lmhash, s, f
      local lmdir
      if params.mobius then
	 lmdir = "/home/runuser/corpus/pubMedCorpora/request1"
      else
	 lmdir = "lm"
      end
      lmhash = loadhash(lmdir .. "/target_words.txt")
      s = string.format(lmdir .. '/words_%s.txt', params.wfsz)
      --s = string.format(lmdir .. '/words_%s.bin', params.wfsz)
      f = torch.DiskFile(s)
      
      print(string.format('[lm] %d words in the lm -- vs %d in the vocabulary', #lmhash, #wordhash))
      print("loading " .. s)
      if false then
	 local toto = f:readFloat(#lmhash * params.wfsz)
	 local fbis = torch.DiskFile(string.format('/home/joel/mobius/pubMedCorpora/request1/words_%s.bin', params.wfsz), "w")
	 fbis:writeObject(toto)
	 fbis:close()
	 exit()
      end
      
      local lmrepr = torch.Tensor(f:readFloat(#lmhash * params.wfsz), 1, #lmhash, -1, params.wfsz, -1)
      --local lmrepr = torch.Tensor(f:readObject(), 1, #lmhash, -1, params.wfsz, -1)
      
      if params.norm then
	 print("normalising embeddings")
	 local mean = lmrepr:mean()
	 local std = lmrepr:std()
	 lmrepr:add(-mean)
	 lmrepr:div(std)
      end
      f:close()
      
      local nknownword = 0

      for wordidx=1, params.nword do
	 local wrd = wordhash[wordidx]
	 if not wrd then
	    print(wordidx)
	    print(#wordhash)
	    print(wordhash[wordidx+1])
	 end
	 while wrd:match('%d%.%d') do
	    wrd = wrd:gsub('%d%.%d', '0')
	 end
	 while wrd:match('%d,%d') do
	    wrd = wrd:gsub('%d,%d', '0')
	 end
	 wrd = wrd:gsub('%d', '0')
	 local lmidx = lmhash[wrd]
	 if lmidx then
	    words.weight[wordidx]:copy(lmrepr[lmidx])
	    nknownword = nknownword + 1
	 end
      end
      print(string.format('[lm] %d known words (over %d in the vocabulary)', nknownword, params.nword))
      print('done')
      collectgarbage()
      
      checklm(words.weight, datas[1].wordhash, params)
      
   end
   
   local network
   if params.arch==1 then --cnn archinecture
      network = {}

      local net = nn.Sequential()
      local features = nn.ParallelTable()
      local fsz = params.wfsz + params.efsz + params.tfsz + params.dtfsz
      features:add(words)
      if params.tfsz~=0 then features:add(entitytags) end
      features:add(entities)
      net:add(features)
      net:add( nn.JoinTable( 2 ) )
      if params.dropout~=0 then
	 local dropout = nn.Dropout(params.dropout)
	 net:add(dropout)
	 net.dropout = dropout
      end
      net:add(nn.TemporalConvolution(fsz, params.nhu[1], params.wsz))
      net:add(nn.HardTanh())--non linearity
      net:add(nn.Max(1))--max pooling
      network.network = net

      network.scorers = {}
      for i=1,#datas do
	 network.scorers[datas[i].corpus] = nn.Sequential()
	 network.scorers[datas[i].corpus]:add(nn.Linear(params.nhu[1], #datas[i].relationhash))
	 network.scorers[datas[i].corpus]:add(nn.LogSoftMax())
      end
      
      function network:forward(input, corpus)
	 self.rep = self.network:forward(input)
	 return self.scorers[corpus]:forward(self.rep)
      end
      
      function network:backward(input, corpus, grad)
	 local gradrep = self.scorers[corpus]:backward(self.rep, grad)
	 self.network:backward(input, gradrep)
      end

      function network:backwardUpdate(input, corpus, grad, lr)
	 local gradrep = self.scorers[corpus]:backwardUpdate(self.rep, grad, lr)
	 self.network:backwardUpdate(input, gradrep, lr)
      end

      
      function network:zeroGradParameters()
	 self.network:zeroGradParameters()
	 for key, scorer in pairs(self.scorers) do
	    scorer:zeroGradParameters()
	 end
      end

      function network:updateParameters(lr)
       	 self.network:updateParameters(lr)
	 for key, scorer in pairs(self.scorers) do
	    scorer:updateParameters(lr)
	 end
      end

      function network:training()
	 self.network:training()
	 for key, scorer in pairs(self.scorers) do
	    scorer:training()
	 end
      end
      
      function network:evaluate()
	 self.network:evaluate()
	 for key, scorer in pairs(self.scorers) do
	    scorer:evaluate()
	 end
      end

      
   elseif params.arch==2 then
      network = nn.Sequential()

      local features = nn.ParallelTable()
      local fsz = params.wfsz + params.efsz + params.tfsz
      features:add(words)
      features:add(entities)
      
      network:add(features)
      network:add(nn.JoinTable( 3 ))
      if params.dropout~=0 then
	 local dropout = nn.Dropout(params.dropout)
	 network:add(dropout)
	 network.dropout = dropout
      end
      network:add(nn.Bottle(nn.Linear(fsz, params.nhu[1])))
      network:add(nn.SplitTable(2))

      -- rnn layers
      local stepmodule = nn.Sequential() -- applied at each time-step
      local inputsize = params.nhu[1]

      for i,hiddensize in ipairs(params.nhu) do
      	 local rnn
      	 if params.rnn=="gru" then -- Gated Recurrent Units
      	    rnn = nn.GRU(inputsize, hiddensize, params.msize, (params.dp==2 or params.dp==3) and params.dropout or nil) --last param: params.dropout/2
      	 elseif params.rnn=="lstm" then -- Long Short Term Memory units
      	    require 'nngraph'
      	    nn.FastLSTM.usenngraph = true -- faster
      	    --nn.FastLSTM.bn = opt.bn
      	    rnn = nn.FastLSTM(inputsize, hiddensize)
      	 elseif params.rnn=="srnn" then -- simple recurrent neural network
      	    local rm =  nn.Sequential() -- input is {x[t], h[t-1]}
      	       :add(nn.ParallelTable()
      		       :add(i==1 and nn.Identity() or nn.Linear(inputsize, hiddensize)) -- input layer
      		       :add(nn.Linear(hiddensize, hiddensize))) -- recurrent layer
      	       :add(nn.CAddTable()) -- merge
      	       :add(nn.Sigmoid()) -- transfer
      	    rnn = nn.Recurrence(rm, hiddensize, 1, 5)
	 else
      	    error("unknown rnn")
      	 end
	 
      	 stepmodule:add(rnn)
      	 -- if opt.dropout > 0 then
      	 --    stepmodule:add(nn.Dropout(opt.dropout))
      	 -- end
      	 inputsize = hiddensize
      end

      -- output layer
      stepmodule:add(nn.Linear(inputsize, params.nhu[1]))
      
      -- encapsulate stepmodule into a Sequencer
      network:add(nn.Sequencer(stepmodule))
      network:add(nn.JoinTable(1))
      network:add(nn.Max(1))--max pooling
      error("adapt to multiple dataset")
      network:add(nn.Linear(params.nhu[1], #data.relationhash))--scorer
      network:add(nn.LogSoftMax())
      network:training()
      
   elseif params.arch==3 then --dependency trees with rnn
      require "nngraph"

      local fsz = params.wfsz + params.efsz + params.tfsz + params.pfsz + (2*params.rdfsz) + params.dtfsz
      
      local getRNN, zeroRNN
      local RNNs
      do
	 RNNs = {}
	 local nClone
	 local cloneCount
	 local sizeMax = math.min(params.maxsize,150)
	 if params.rnn=="linear" then
	    nClone = {}
	    cloneCount = {}
	    for i=1,sizeMax do nClone[i]=2 end
	    for i=1,sizeMax do
	       RNNs[i]={nn.Sequential():add(nn.Linear(params.nhu[1] * i, params.nhu[1])):add(nn.HardTanh())}
	       for j=2,nClone[i] do
		  RNNs[i][j] = RNNs[i][1]:clone("weight", "bias")
	       end
	    end
	 elseif params.rnn=="gru" then
	    nClone = 60
	    local insize = params.nhu[1] + (params.dtfsz>0 and params.dt==2 and params.dtfsz or 0)
	    print("insize " .. insize)
	    local drop = (params.dp==2 or params.dp==3) and params.dropout or nil
	    local gru = nn.GRU(insize , params.nhu[1], params.msize, drop) 
	    gru = nn.Sequencer(gru)
	    RNNs[1] = nn.Sequential():add(gru):add(nn.SelectTable(-1))
	    for i=2,nClone do RNNs[i] = RNNs[1]:clone("weight", "bias") end
	 elseif params.rnn=="bgru" then
	    nClone = 50
	    local insize = params.nhu[1] + (params.dtfsz>0 and params.dt==2 and params.dtfsz or 0)
	    local drop = (params.dp==2 or params.dp==3) and params.dropout or nil
	    local gru1 = nn.GRU(insize , params.nhu[1], params.msize, drop) 
	    gru1 = nn.Sequencer(gru1)
	    local gru2 = nn.GRU(insize , params.nhu[1], params.msize, drop)
	    gru2 = nn.Sequencer(gru2)

	    local conc = nn.ConcatTable()
	    conc:add( nn.Sequential():add(gru1):add(nn.SelectTable(-1)) )
	    conc:add( nn.Sequential():add(nn.ReverseTable()):add(gru2):add(nn.SelectTable(-1)) )

	    RNNs[1] = nn.Sequential():add(conc):add(nn.CAddTable())
	    
	    for i=2,nClone do RNNs[i] = RNNs[1]:clone("weight", "bias") end
	 elseif params.rnn=="lstm" then
	    nClone = 50
	    if params.dropout~=0 and (params.dp==2 or params.dp==3) then error("not implemented yet") end
	    nn.FastLSTM.usenngraph = true -- faster
      	    local insize = params.nhu[1] + (params.dtfsz>0 and params.dt==2 and params.dtfsz or 0)
	    --nn.FastLSTM.bn = opt.bn
      	    local lstm = nn.FastLSTM(insize, params.nhu[1], params.msize)
	    lstm = nn.Sequencer(lstm)
	    RNNs[1] = nn.Sequential():add(lstm):add(nn.SelectTable(-1))
	    for i=2,nClone do RNNs[i] = RNNs[1]:clone("weight", "bias") end
	 elseif params.rnn=="srnn" then
	    error("")
	 elseif params.rnn=="cnn" then
	    nClone = 50
	    if params.dropout~=0 then error("not implemented yet") end
	    nn.FastLSTM.usenngraph = true -- faster
      	    local insize = params.nhu[1] + (params.dtfsz>0 and params.dt==2 and params.dtfsz or 0)
	    local cnn = nn.Sequential():add(nn.JoinTable(1)):add(nn.TemporalConvolution(insize, params.nhu[1], params.wsz2)):add(nn.HardTanh()):add(nn.Max(1)):add(nn.Unsqueeze(1))
	    RNNs[1] = cnn
	    for i=2,nClone do RNNs[i] = RNNs[1]:clone("weight", "bias") end
	 else
	    error("")
	 end
	 
	 getRNN = function(params, size)
	    local rnn = nn.Sequential()
	    if params.rnn=="linear" then
	       if size then
		  rnn:add(nn.JoinTable(1))
		  --add more clones
		  if cloneCount[size]>nClone[size] then
		     nClone[size] = nClone[size] + 1
		     RNNs[size][ nClone[size] ] = RNNs[size][1]:clone("weight", "bias")
		     --print("adding new clone of size " .. size)
		     --io.read()
		  end
		  rnn:add( RNNs[size][ cloneCount[size] ])
		  cloneCount[size] = cloneCount[size] + 1
	       else
		  rnn:add(nn.CAddTable())
	       end
	    elseif params.rnn=="gru" then
	       rnn:add(RNNs[cloneCount])
	       cloneCount = cloneCount + 1
	       if cloneCount>#RNNs then print(cloneCount); print(#RNNs); error("not enough clones") end
	       if false then
		  local input = {torch.rand(1,200), torch.rand(1,200), torch.rand(1,200), torch.rand(1,200), torch.rand(1,200)}
		  print(input); print(rnn)
		  local output = rnn:forward(input)
		  print(output); exit()
	       end
	    elseif params.rnn=="bgru" then
	       rnn:add(RNNs[cloneCount])
	       cloneCount = cloneCount + 1
	       if cloneCount>#RNNs then print(cloneCount); print(#RNNs); error("not enough clones") end
	       if false then
		  local input = {torch.rand(1,200), torch.rand(1,200), torch.rand(1,200), torch.rand(1,200), torch.rand(1,200)}
		  print(input); print(rnn)
		  local output = rnn:forward(input)
		  print(output); exit()
	       end
	    elseif params.rnn=="lstm" then
	       rnn:add(RNNs[cloneCount])
	       cloneCount = cloneCount + 1
	       if cloneCount>#RNNs then print(cloneCount); print(#RNNs); error("not enough clones") end
	       if false then
		  local input = {torch.rand(1,200), torch.rand(1,200), torch.rand(1,200), torch.rand(1,200), torch.rand(1,200)}
		  print(input); print(rnn)
		  local output = rnn:forward(input)
		  print(output); exit()
	       end
	    elseif params.rnn=="cnn" then
	       rnn:add(RNNs[cloneCount])
	       cloneCount = cloneCount + 1
	       if false then
		  local input = {torch.rand(1,200), torch.rand(1,200), torch.rand(1,200), torch.rand(1,200), torch.rand(1,200)}
		  print(input); print(rnn)
		  local output = rnn:forward(input)
		  print(output); exit()
	       end
	       if cloneCount>#RNNs then print(cloneCount); print(#RNNs); error("not enough clones") end
	    else
	       error("")
	    end
	    
	    return rnn
	 end
	 zeroRNN = function()
	    if params.rnn=="linear" then
	       for i=1,sizeMax do
		  cloneCount[i] = 1
	       end
	    elseif params.rnn=="gru" or params.rnn=="bgru" or params.rnn=="lstm" or params.rnn=="srnn" or params.rnn=="cnn" then
	       cloneCount = 1
	    else 
	       error("")
	    end
	 end
      end
      
      local _lookup = nn.Sequential()
      local dropout = {}
      local par = get_par(params, lkts, dropout)
      
      -- local input = torch.Tensor({1,2,3,4,5,6})
      -- local input2 = torch.Tensor({1,2,3,4,5,6})
      -- _lookup:forward({input, input2})
      
      local _lookup2
      if params.dtfsz~=0 and params.dt==2 then
	 _lookup2 = nn.Sequential():add(nn.LookupTable(#datas[1].deptypehash, params.dtfsz))
	 if params.dropout~=0 and params.dp~=2 then
	    local d = nn.Dropout(params.dropout)
	    table.insert(dropout, d)
	    _lookup2:add(d)
	 end
	 _lookup2:add(nn.SplitTable((params.dropout and params.dp==2) and 2 or 1)) --potential bug
      end
      
      _lookup:add(par)
      _lookup:add(nn.JoinTable((params.batch~=0 or params.dp==2 or params.dp==3 or params.rnn=="lstm" or params.rnn=="cnn") and 3 or 2))
      --local l = nn.Sequential():add(nn.Linear(fsz, params.nhu[1])):add(nn.HardTanh())
      local l = nn.Sequential():add(nn.TemporalConvolution(fsz, params.nhu[1],params.wsz)):add(nn.HardTanh())
      if params.batch~=0 or params.dp==2 or params.dp==3 or params.rnn=="lstm" or params.rnn=="cnn" then
	 l = nn.Bottle(l)
	 _lookup:add(l)
      else
	 _lookup:add(l)
      end

      
      if false then
	 local input = {torch.Tensor({{1,2,3}}), torch.Tensor{{2,3,4}}}
	 print(par)
	 print(_lookup)
	 print(input)

	 local output = par:forward(input)
	 print("===")
	 print(output)
	 print("===")

	 local output = _lookup:forward(input)
	 print("===")
	 print(output)
	 print("===")
	 exit()
      end

      
      network = {}
      
      network.scorers = {}
      for i=1, #datas do
	 local _scorer = nn.Sequential()
	 _scorer:add(nn.Linear(params.nhu[1], #datas[i].relationhash))
	 _scorer:add(nn.LogSoftMax())
	 network.scorers[datas[i].corpus] = _scorer
      end
	 
      network.save = {}
      table.insert(network.save, _lookup)
      table.insert(network.save, _lookup2)
      table.insert(network.save, RNNs[1])
      table.insert(network.save, _scorer)
      network.RNNs = RNNs
      network.dropout = dropout
      if params.dropout~=0 and params.dp~=2 then
	 function network.dropout:training()
	    for i=1,#self do self[i]:training() end
	 end
	 function network.dropout:evaluate()
	    for i=1,#self do self[i]:evaluate() end
	 end
      end
      network.lookup = _lookup
      network.lookup2 = _lookup2
      
      local g
      function network.getGraph(self, graphTab, data)
	 
	 zeroRNN()
	 
	 local lookup = _lookup()
	 local lookup2
	 if params.dtfsz~=0 and params.dt==2 then
	    lookup2 = _lookup2()
	 end
	 
	 local _size = 0
	 local nrep = 0--graphTab[1]
	 local tab = {}
	 local i=1
	 while i<=#graphTab do
	    nrep = nrep + 1
	    local size = graphTab[i]
	    _size = _size + size
	    i = i+1
	    local t = {}
	    for j=1,size do
	       t[j]=graphTab[i]
	       i = i + 1
	    end
	    table.insert(tab, t)
	 end
	 _size = _size - nrep +1

	 --print(graphTab)
	 --print(tab)
	 --print(nrep)

	 local root = tab[#tab][1]
	 --print("root " .. root)
	 
	 local split = nn.SplitTable((params.batch~=0 or params.dp==2 or params.dp==3 or params.rnn=="lstm" or params.rnn=="cnn") and 2 or 1)(lookup)
	 
	 --print(_size)
	 
	 local inputs = {}
	 for i=1, _size do
	    inputs[i] = nn.SelectTable(i)(split):annotate{name = 'in' .. i}
	 end
	 local inputsdt = {}
	 if params.dtfsz~=0 and (params.dt==2 or params.dp==3) then
	    for i=1,_size do
	       inputsdt[i] = nn.SelectTable(i)(lookup2):annotate{name = "indt" .. i}
	    end
	 end
	 
	 
	 local RNNs = {}
	 if true then
	    for i=1,nrep do
	       local t,tdt = {}, {}
	       for j=1,#tab[i] do --graph links
		  if tab[i][j]<1000 then --word
		     if params.dtfsz~=0 and params.dt==2 then
			if j==1 then --head of the relation
			   table.insert(tdt, inputsdt[ root ])
			   --print(1)
			else --not head
			   table.insert(tdt, inputsdt[ tab[i][j] ])
			   --print(tab[i][j])
			end
		     end
		     table.insert(t, inputs[ tab[i][j] ])
		  else --representation
		     if params.dtfsz~=0 and params.dt==2 then
			if j==1 then --head
			   table.insert(tdt, inputsdt[ 1 ])
			   print("je ne dois pas passer par la...")
			   exit()
			else
			   --print( tab[ tab[i][j]-1000 ][1] )
			   table.insert(tdt, inputsdt[ tab[ tab[i][j]-1000 ][1] ])
			end
		     end
		     table.insert(t, RNNs[ tab[i][j]-1000 ])
		  end
	       end
	       local t2
	       if params.dtfsz~=0 and params.dt==2 then
		  t2 = {}
		  for i=1,#t do
		     table.insert(t2, nn.JoinTable((params.dropout~=0 and params.dp==2) and 2 or 1)({t[i], tdt[i]}))
		  end
		  t = t2
	       end
	       --print(t)
	       --print(t2)
	       --io.read()
	       RNNs[i] = getRNN(params, #tab[i])
	       RNNs[i] = RNNs[i](t):annotate({name = "RNN" .. i })
	    end
	 end
	 

	 --scorer = _scorer(RNNs[#RNNs]) --back
	 scorer = self.scorers[data.corpus](RNNs[#RNNs])
	 
	 g = nn.gModule({lookup,lookup2},{scorer})
	 if params.display then
	    graph.dot(g.fg, 'Forward Graph',"mygraph")   
	    io.read()
	 end
	 
	 network.g = g
      end

      function network:forward(input)
	 return g:forward(input)
      end

      function network:backward(input, grad)
	 return g:backward(input, grad)
      end
      
      function network:updateParameters(lr)
	 return g:updateParameters(lr)
      end

      
      function network:zeroGradParameters()
	 g:zeroGradParameters()
      end

      function network:training()
	 for i=1,#self.RNNs do
	    self.RNNs[i]:training()
	 end
      end
      
      function network:evaluate()
	 for i=1,#self.RNNs do
	    self.RNNs[i]:evaluate()
	 end
      end
      
   elseif params.arch==4 or params.arch==5 then
      -- share module parameters
      function share_params(cell, src)
	 if torch.type(cell) == 'nn.gModule' then
	    for i = 1, #cell.forwardnodes do
	       local node = cell.forwardnodes[i]
	       if node.data.module then
		  node.data.module:share(src.forwardnodes[i].data.module,
					 'weight', 'bias', 'gradWeight', 'gradBias')
	       end
	    end
	 elseif torch.isTypeOf(cell, 'nn.Module') then
	    cell:share(src, 'weight', 'bias', 'gradWeight', 'gradBias')
	 else
	    error('parameters cannot be shared for this input')
	 end
      end

      network = {}

      local treelstm_config = {
	 in_dim = params.nhu[1],
	 mem_dim = params.nhu[1],
	 gate_output = params.gateoutput,
	 dropout = (params.dropout~=0 and params.dp==2 or params.dp==3)
	    and params.dropout or nil,
	 optim = params.optim
      }
      if params.arch==4 then
	 network.treelstm = treelstm.ChildSumTreeLSTM(treelstm_config)
      else
	 treelstm_config.tags={1,2}--one for not in the shortest path
	 network.treelstm = treelstm.DTTreeLSTM(treelstm_config)
      end
      
      local fsz = params.wfsz + params.efsz + params.tfsz + params.pfsz + (2*params.rdfsz) + params.dtfsz
      
      network.lookup = nn.Sequential()
      local dropout = {}
      local par = get_par(params, lkts, dropout)
      
      network.lookup:add(par)
      network.lookup:add(nn.JoinTable(2))
      local l = nn.Sequential():add(nn.TemporalConvolution(fsz, params.nhu[1],params.wsz)):add(nn.HardTanh())
      network.lookup:add(l)
      
      network.scorers = {}
      for i=1, #datas do
	 network.scorers[datas[i].corpus] = nn.Sequential()
	 network.scorers[datas[i].corpus]:add(nn.Linear(params.nhu[1], #datas[i].relationhash))
	 local d
	 if params.dropout~=0 and (params.dp==3 or params.dp==4) then
	    d = nn.Dropout(params.dropout)
	    table.insert(dropout, d)
	    network.scorers[datas[i].corpus]:add(d)
	 end
	 network.scorers[datas[i].corpus]:add(nn.LogSoftMax())
      end

      network.save = {}
      table.insert(network.save, network.lookup)
      table.insert(network.save, network.treelstm)
      table.insert(network.save, network.scorers)
      
      network.dropout = dropout
      if params.dropout~=0 and params.dp~=2 then
	 function network.dropout:training()
	    for i=1,#self do self[i]:training() end
	 end
	 function network.dropout:evaluate()
	    for i=1,#self do self[i]:evaluate() end
	 end
      end
      
      local zeros = torch.zeros(params.nhu[1])
      function network:forward(tree, input, corpus)
	 self.emb = self.lookup:forward(input)
	 self.rep = self.treelstm:forward(tree, self.emb)[2]
	 return self.scorers[corpus]:forward(self.rep)
      end
      
      function network:backward(tree, input, corpus, grad)
	 local gradrep = self.scorers[corpus]:backward(self.rep, grad)
	 local grademb = self.treelstm:backward(tree, self.emb, {zeros,gradrep})
	 self.lookup:backward(input, grademb)
      end
      
      function network:zeroGradParameters()
	 self.lookup:zeroGradParameters()
	 self.treelstm:zeroGradParameters()
	 for key, scorer in pairs(self.scorers) do
	    scorer:zeroGradParameters()
	 end
      end

      function network:updateParameters(lr)
       	 self.lookup:updateParameters(lr)
	 self.treelstm:updateParameters(lr)
	 for key, scorer in pairs(self.scorers) do
	    scorer:updateParameters(lr)
	 end
      end

      function network:training()
	 self.treelstm:training()
      end
      
      function network:evaluate()
	 self.treelstm:evaluate()
      end
      
   elseif params.arch==6 then --multi chanel cnn
      if params.dp==2 or params.dp==3 then error("dp==2 not authorized for arch==6") end

      local wszs = {}
      network = nn.Sequential()
      local net = nn.ConcatTable()
      local fsz = params.wfsz + params.efsz + params.tfsz + params.pfsz + (2*params.rdfsz) + params.dtfsz
      local dropout = {}
      for i=1,#params.wszs do
	 local pad = (params.wszs[i]-1)/2
	 wszs[i] = nn.Sequential()
	 local padding = nn.MapTable()
	 local p = nn.Sequential()
	 p:add(nn.Padding(1,-pad,1,1))
	 p:add(nn.Padding(1,pad,1,1))
	 padding:add(p)
	 wszs[i]:add(padding)

	 if false then
	    exit()
	    wszs[i]:add( get_par(params, lkts, dropout) )
	    wszs[i]:add(nn.JoinTable(2))
	 else
	    local channels = nn.ConcatTable()
	    channels:add( nn.Sequential():add( get_par(params, lkts, dropout)):add(nn.JoinTable(2))  )
	    if params.channels>1 then
	       	    channels:add( nn.Sequential():add( get_par(params, lkts, dropout, true)):add(nn.JoinTable(2))  )
	    end
	    if params.channel>2 then error("not implemented") end
	    wszs[i]:add(channels)
	    wszs[i]:add( nn.CAddTable() )
	 end

	 wszs[i]:add( nn.TemporalConvolution(fsz, params.nhu[1], params.wszs[i]) )
	 wszs[i]:add( nn.HardTanh() )--non linearity
	 wszs[i]:add( nn.Max(1) )
	 net:add(wszs[i])
      end
      
      network:add(net)
      network:add( nn.JoinTable(1))

      --print(network)
      --exit()
      
      network = {network=network}
           
      network.scorers = {}
      for i=1,#datas do
	 network.scorers[datas[i].corpus] = nn.Sequential()
	 network.scorers[datas[i].corpus]:add(nn.Linear(#wszs * params.nhu[1], #datas[i].relationhash))
	 network.scorers[datas[i].corpus]:add(nn.LogSoftMax())
      end

      network.dropout = dropout
      if params.dropout~=0 and params.dp~=2 then
	 function network.dropout:training()
	    for i=1,#self do self[i]:training() end
	 end
	 function network.dropout:evaluate()
	    for i=1,#self do self[i]:evaluate() end
	 end
      end
      
      function network:forward(input, corpus)
	 self.rep = self.network:forward(input)
	 return self.scorers[corpus]:forward(self.rep)
      end
      
      function network:backward(input, corpus, grad)
	 print("grad")
	 print(grad)
	 local gradrep = self.scorers[corpus]:backward(self.rep, grad)
	 self.network:backward(input, gradrep)
      end

      function network:backwardUpdate(input, corpus, grad, lr)
	 local gradrep = self.scorers[corpus]:backwardUpdate(self.rep, grad, lr)
	 self.network:backwardUpdate(input, gradrep, lr)
      end

      function network:zeroGradParameters()
	 self.network:zeroGradParameters()
	 for key, scorer in pairs(self.scorers) do
	    scorer:zeroGradParameters()
	 end
      end

      function network:updateParameters(lr)
       	 self.network:updateParameters(lr)
	 for key, scorer in pairs(self.scorers) do
	    scorer:updateParameters(lr)
	 end
      end

      function network:training()
	 self.network:training()
	 for key, scorer in pairs(self.scorers) do
	    scorer:training()
	 end
      end
      
      function network:evaluate()
	 self.network:evaluate()
	 for key, scorer in pairs(self.scorers) do
	    scorer:evaluate()
	 end
      end
      
      network.save = {}
      table.insert(network.save, network.network)
      table.insert(network.save, network.scorers)
      
      
      if false then
	 local input = {torch.Tensor({4,2,3}), torch.Tensor({4,2,3})}
	 print("net")
	 print(net)
	 print("input")
	 print(input)
	 local output = network:forward(input, "full")
	 print("output")
	 print(output)
	 -- print(output[1][1])
	 -- print(output[1][2])
	 -- print(output[2][1])
	 -- print(output[2][2])
	 exit()
      end
   else
      error("unknown arch")
   end


   function network:printnet()
      
   end
   
   function network:getnetsave(params)
      if params.arch==1 or params.arch==2 then
	 --return network:clone("weight", "bias")
	 return nil
      elseif params.arch==4 or params.arch==5 then
	 local res = {}
	 for i=1,2 do
	    local p, g = network.save[i]:parameters()--clone("weight", "bias"):
	    --if i==2 then p = g end
	    table.insert(res, p)
	 end
	 res[3] = {}
	 for k,v in pairs(network.scorers) do
	    res[3][k] = v:clone("weight", "bias"):parameters()
	 end
	 return res
      elseif params.arch==6 then
	 local res = {}
	 for i=1,1 do
	    local p, g = network.save[i]:parameters()--clone("weight", "bias"):
	    table.insert(res, p)
	 end
	 res[2] = {}
	 for k,v in pairs(network.scorers) do
	    res[2][k] = v:clone("weight", "bias"):parameters()
	 end
	 return res
      else
	 local res = {}
	 print(#network.save)
	 for i=1,#network.save do
	    print(i)
	    print(network.save[i])
	    local p = network.save[i]:clone("weight", "bias"):parameters()
	    table.insert(res, p)
	 end
	 return res
      end
   end
   
   function network:loadnet(params, net)
      local c=1
      if params.arch==1 or params.arch==2 then
	 error("to do")
      elseif params.arch==4 or params.arch==5 then
	 --print(net)
	 for i=1,2 do
	    local oldparameters, oldgrads = network.save[i]:parameters()
	    --if i==2 then oldparameters = oldgrads end
	    for j=1,#oldparameters do
	       oldparameters[j]:copy( net[i][j] )
	    end
	 end
	 for k,v in pairs(network.scorers) do
	    --print(k)
	    local oldparameters = v:parameters()
	    --print(oldparameters)
	    for j=1,#oldparameters do
	       oldparameters[j]:copy( net[3][k][j] )
	    end
	 end
      elseif params.arch==6 then
	 --print(net)
	 for i=1,1 do
	    local oldparameters, oldgrads = network.save[i]:parameters()
	    --if i==2 then oldparameters = oldgrads end
	    for j=1,#oldparameters do
	       oldparameters[j]:copy( net[i][j] )
	    end
	 end
	 for k,v in pairs(network.scorers) do
	    --print(k)
	    local oldparameters = v:parameters()
	    --print(oldparameters)
	    for j=1,#oldparameters do
	       oldparameters[j]:copy( net[2][k][j] )
	    end
	 end
      else
	 for i=1,#network.save do
	    local oldparameters = network.save[i]:parameters()
	    for j=1,#oldparameters do
	       oldparameters[j]:copy( net[i][j] )
	    end
	 end
      end
   end
   
   return network
end
