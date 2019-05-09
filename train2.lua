require('data2')
require('torch')
require('nn')
require('rnn')
require('network')
require('test2')
require("trepl")
require("nngraph")

function printw(t, dict)
   print(t:size(1))
   for i=1,t:size(1) do
      io.write(dict[t[i]] .. " ")
   end
   io.write("\n")
end

function printinput(t, dict, ent)
   for i=1,t:size(1) do
      local w = dict[t[i]]
      if ent[i]~=2 then w = w:upper() end
      io.write(w .. " ")
   end
   io.write("\n")
end

cmd = torch.CmdLine()

cmd:text()
cmd:text('Chunk-based phrase prediction')
cmd:text()
cmd:text()
cmd:text('Misc options:')
cmd:option('-optim', false, 'cdlinear optim')
cmd:option('-l1norm', 0, 'l1 norm rate')
cmd:option('-l2norm', 0, 'l2 norm rate')
cmd:option('-lnormpos', '', 'position for l2norm (emb, model, scorer)')
cmd:option('-lnormnobias', false, 'do not regularize bias')
cmd:option('-gateoutput', false, 'option for treelstm')
cmd:option('-validp', 10, 'training corpus proportion for the validation')
cmd:option('-valids', 1, 'sector extracted from training for the validation')
cmd:option('-truesgd', false, 'true sgd (not sentence by sentence sgd')
cmd:option('-balance', false, 'adapte learning rate according to relation distribution')
cmd:option('-wfsz', 100, 'word feature size')
cmd:option('-efsz', 10, 'entity feature size (for the 2 candidate entities)')
cmd:option('-tfsz', 0, 'entity tag features size')
cmd:option('-pfsz', 0, 'pos tag features size')
cmd:option('-rdfsz', 0, 'relative distance features size')
cmd:option('-dtfsz', 0, 'dependency type feature size')
cmd:option('-dt', 1, 'dependency type method: 1=concatenate tag to child/2=concatenate tag do child except for the head (head tag)')
cmd:option('-wsz', 1, 'window size')
cmd:option('-wsz2', 1, 'window size for rnn-cnn')
cmd:option("-nword", 20000, "dictionary size")
cmd:option("-nhu", '{200}', "hidden units")
cmd:option("-arch", 1, "nn architecture")
--cmd:option("-")
cmd:option("-rnn", "", "rnn to use for arch 2 or arch 3")
cmd:option("-msize", 5, 'memory size for gru')
cmd:option('-seed', 1111, 'seed')
cmd:option('-lr', 0.001, 'learning rate')
cmd:option('-lm', false, 'use language model')
cmd:option('-norm', false, 'normalise lm')
cmd:option('-log', false, "log file")
cmd:option('-restart', '', 'model to restart from')
cmd:option('-dir', '.', 'subdirectory to save the stuff')
cmd:option('-dropout', 0, 'add dropout')
cmd:option('-dp', 0, 'dropout position 1:features / 2:gru / 4:scorer / 3:all')
cmd:option('-maxload', math.huge, 'data to load')
cmd:option('-maxsize', math.huge, 'sentencesizemax')
cmd:option('-display', false, 'display network')
cmd:option('-notest', false, 'do not test')
cmd:option('-notestcorpus', false, 'do not extract test')
cmd:option('-debug', false, 'debug option for nngraph')
cmd:option('-mobius', false, 'run on mobius')
cmd:option('-corpus', '{full}', 'corpus to use')
cmd:option('-parser', 'stanford', 'parser to use (stanford or McClosky)')
cmd:option('-nosgd', false, 'no sgd')
cmd:option('-time', false, 'time evaluation')
cmd:option('-batch', 0, 'batch sentence per sentence')
cmd:option('-batchdiv', false, 'div lr by batch size')
cmd:option('-anon', false, 'anonymize drugs')
cmd:option('-debug2', false, 'debug2')
cmd:option('-maxent', 1000, 'max entities in training sentence')
cmd:option('-restartparams', '{}', 'max entities in training sentence')
cmd:option('-niter', 100, 'max iter')
cmd:option('-crossbalance', 0, 'cross corpus balance')
cmd:option('-testc', 0, 'corpus to test on')
cmd:option('-wszs', '{3,3,5,5}', 'corpus to test on')
cmd:option('-channels', 1, '')
cmd:option('-1r', 0, 'use only 1 specified relation for the additional corpus') --only for 1 additional corpus
cmd:option('-fusr', '{}', 'fusion relations for the additional corpus')
cmd:option('-select_data', '{}', 'fusion relations for the additional corpus')
cmd:option('-mergerelations', 0, 'merge relations: 1=onerelation, 2=keep isEquivalentTo')
cmd:option('-aotf', false, 'anonymize on the fly')
cmd:option('-hierarchy', 0, "pk_phenotype -> phenotype, ... for input entiites")
cmd:option('-decodedir', '', '')
cmd:text()

local params = cmd:parse(arg)

torch.setdefaulttensortype('torch.FloatTensor')

torch.setnumthreads(1)

if true then
   torch.manualSeed(params.seed)
else
   torch.manualSeed(os.time())
end


params.corpus = params.corpus:gsub("{", "{\"")
params.corpus = params.corpus:gsub("}", "\"}")
params.corpus = params.corpus:gsub(" +", "")
params.corpus = params.corpus:gsub(",", "\",\"")
--params.select_data = params.select_data:gsub("{", "{\"")
--params.select_data = params.select_data:gsub("}", "\"}")
params.select_data = params.select_data:gsub(" +", "")
--params.select_data = params.select_data:gsub(",", "\",\"")


if params.fusr~="{}" then
   params.fusr = params.fusr:gsub("{", "{\"")
   params.fusr = params.fusr:gsub("}", "\"}")
   params.fusr = params.fusr:gsub(" +", "")
   params.fusr = params.fusr:gsub(",", "\",\"")
end
params.nhu = loadstring("return " .. params.nhu)()
params.corpus = loadstring("return " .. params.corpus)()
--print(params.select_data)
params.select_data = loadstring("return " .. params.select_data)()
--print(params.select_data)
--exit()

params.fusr = loadstring("return " .. params.fusr)()
params.wszs = loadstring("return " .. params.wszs)()  
local restartparams = loadstring("return " .. params.restartparams)()

if params.arch==6 and params.wsz~=1 then error("") end

local frestart
local rundir
local expidx = 0
if params.restart ~= '' then
   print(string.format('restarting from <%s>', params.restart))
   frestart = torch.DiskFile(params.restart):binary()
   local decodedir = params.decodedir
   local mobius = params.mobius
   local restart = params.restart
   params = frestart:readObject()
   params.restart = restart
   rundir = params.rundir
   for i=0,99 do
      expidx = i
      local fname = string.format('%s/log-%.2d', rundir, expidx)
      local f = io.open(fname)
      if f then
         print(string.format('<%s> found', fname))
         f:close()
      else
         break
      end
   end
   params.decodedir = decodedir
   params.mobius = mobius
   --params.corpus = params.corpus or 'drugbank'
   if not params.pfsz then params.pfsz=0 end
   if not params.tfsz then params.tfsz=0 end
   if not params.rdfsz then params.rdfsz=0 end

   for k,v in pairs(restartparams) do
      params[k] = v
   end
else
   rundir = cmd:string('exp', params, {dir=true, nhu=true, corpus=true, select_data=true, restartparams=true, wszs=true, fusr=true})
   rundir = rundir .. ",nhu={" .. params.nhu[1]
   for i=2,#params.nhu do
      rundir = rundir .. "-" .. params.nhu[i]
   end
   rundir = rundir .. "}"
   rundir = rundir .. ",wszs={" .. params.wszs[1]
   for i=2,#params.wszs do
      rundir = rundir .. "-" .. params.wszs[i]
   end
   rundir = rundir .. "}"
   rundir = rundir .. ",corpus={" .. params.corpus[1]
   for i=2,#params.corpus do
      rundir = rundir .. "-" .. params.corpus[i]
   end
   rundir = rundir .. "}"
   --print(params.select_data)
   if #params.select_data~=0 then
      local toto = params.select_data["full"]:match("_([^_]+)$")
      rundir = rundir .. ",select={" .. toto 
      rundir = rundir .. "}"
      if params.fusr[1] then
	 rundir = rundir .. ",fusr={" .. params.fusr[1]
	 for i=2,#params.fusr do
	    rundir = rundir .. "-" .. params.fusr[i]
	 end
      end
      rundir = rundir .. "}"
   end
   if params.dir ~= '.' then
      rundir = params.dir .. '/' .. rundir
   end
   params.rundir = rundir
   print(params.rundir)
   os.execute('mkdir -p ' .. rundir)
   params.currentiter = 0
end
if params.restart=='' and params.log then
   cmd:log(string.format('%s/log-%.2d', rundir, expidx), params)
end

if params.fusr==nil then params.fusr = {} end
--if params.arch==3 and params.wsz>1 then error("wsz must be 1 for arch3") end
if params.batch==false then params.batch=0 end


if params.arch==4 or params.arch==5 then
   treelstm = {}
   include("./treeLSTM/util/Tree.lua")
   include('./treeLSTM/layers/CRowAddTable.lua')
   include('./treeLSTM/models/LSTM.lua')
   include('./treeLSTM/models/TreeLSTM.lua')
   include('./treeLSTM/models/ChildSumTreeLSTM.lua')
   if params.arch==5 then
      include('./treeLSTM/layers/ConditionedLinear.lua')
      include('./treeLSTM/layers/ConditionedLinear2.lua')   
      include('./treeLSTM/models/DTTreeLSTM.lua')
   end
   function treelstm.Tree:print_tree(tab)
      print(tab .. self.idx .. " " .. (self.tag and self.tag or "notag"))
      io.write(tab .. "{")
      for i = 1,#self.sontags do io.write(self.sontags[i] .. " ") end
      io.write("}\n")
      for i=1,#self.children do
	 self.children[i]:print_tree(tab .. "\t")
      end
   end
   function treelstm.Tree:LCA(a, b, entities, default_tag, inpath_tag)
      self.tag=default_tag
      local fa, fb, found
      for i=1,#self.children do
	 _fa, _fb, _found = self.children[i]:LCA(a,b,entities,default_tag, inpath_tag)
	 found = _found or found
	 fa = _fa or fa
	 fb = _fb or fb
      end
      if not found then
	 fa = fa or (entities[self.idx]==a)
	 fb = fb or (entities[self.idx]==b)
	 if fa or fb then self.tag=inpath_tag end
      end
      return fa, fb, (fa and fb)
   end
   function treelstm.Tree:set_tab_sontags()
      self.sontags = {}
      if #self.children==0 then
	 table.insert(self.sontags, self.tag)
      else
	 for i=1,#self.children do
	    table.insert(self.sontags, self.children[i].tag)
	    self.children[i]:set_tab_sontags()
	 end
      end
   end
end

local relation_pos = {}
function is_related(params, data, nsent, ent1, ent2)
   if data.corpus=="medline" or data.corpus=="drugbank" or data.corpus=="full" then
      return true
   elseif data.corpus=="snpphena" then
      local t1 = data.entities.typeent(data, nsent, ent1)
      local t2 = data.entities.typeent(data, nsent, ent2)
      --print(t1 .. " " .. t2 .. " " .. (((t1=="SNP" and t2=="Phenotype") or (t2=="SNP" and t1=="Phenotype")) and "true" or "false"))
      return ((t1=="SNP" and t2=="Phenotype") or (t2=="SNP" and t1=="Phenotype"))
   elseif data.corpus=="ppi" or data.corpus=="LLL" or data.corpus=="AIMed" or data.corpus=="HPRD50" or data.corpus=="IEPA" or data.corpus=="BioInfer" then
      return true
   elseif data.corpus=="ADE" then
      return true
   elseif data.corpus=="reACE" then
      return true
   elseif data.corpus=="EUADR_drug_disease" or data.corpus=="EUADR_drug_target" or data.corpus=="EUADR_target_disease" then
      return true --maybe can be optimized
   elseif data.corpus=="PGxCorpus" then
      local overlap = data.entities.overlap(data, nsent, ent1, ent2)
      if overlap then return false end
      local t1 = data.entities.typeent(data, nsent, ent1)
      local t2 = data.entities.typeent(data, nsent, ent2)

      --print(data.relations:isrelated(nsent, ent1, ent2))
      if data.relations:isrelated(nsent, ent1, ent2)~=1 then
	 if relation_pos[t1 .. "-" .. t2]==nil then relation_pos[t1 .. "-" .. t2]=1 else relation_pos[t1 .. "-" .. t2]= relation_pos[t1 .. "-" .. t2] +1 end
      end
      return true
   else
      error()
   end
end

function _select(params, data, nsent, ent1, ent2)
   --print(nsent .. " " .. ent1 .. " " .. ent2)
   --print(data.select_data[nsent])
   --print(data.corpus)
   --print(data.select_data==nil)
   --print(not (data.select_data~=nil and data.select_data[nsent][ent1]~=nil and data.select_data[nsent][ent1][ent2]~=nil))
   local res = not (data.select_data~=nil and data.select_data[nsent][ent1]~=nil and data.select_data[nsent][ent1][ent2]~=nil)
   --if not res then io.read() end
   return res
end
   
print("restart with : ")
print("/home/joel/torch/install/bin/luajit /home/joel/Bureau/loria/code/semeval2013DDI/train.lua -restart " .. string.format('%s/model.bin', rundir))

local datas = {}
local tdatas = {}
local vdatas = {}
local subtraindatas = {}

for i=1,#params.corpus do
   print(params.corpus[i])
   loadhash(params, params.corpus[i])
end

for i=1,#params.corpus do
   local data = createdata(params, params.corpus[i], "train")

   -- printw(data.words[4], data.wordhash)
   -- print(data.relations[4])
   -- for i=1,#data.entities[4] do
   --    print(data.entities[4][i])
   --    print(data.entities[4][i][2])
   -- end
   -- exit()
   
   table.insert(datas, data)
   
   if params.corpus[i]~="PGxCorpus" and params.corpus[i]~="reACE" and params.corpus[i]~="ppi" and params.corpus[i]~="LLL" and params.corpus[i]~="AIMed" and params.corpus[i]~="HPRD50" and params.corpus[i]~="IEPA" and params.corpus[i]~="BioInfer" and params.corpus[i]~="ADE" and (not params.corpus[i]:match("EUADR")) then
      local tdata = createdata(params, params.corpus[i], "test")
      table.insert(tdatas, tdata)
      if params.validp~=0 then
   	 local vdata = extract_data(datas[i], params.validp, params.valids, true)
   	 table.insert(vdatas, vdata)
      else
   	 table.insert(vdatas, tdata)
      end
      local subtraindata = extract_data(datas[i], params.validp, params.valids, false)
      table.insert(subtraindatas, subtraindata)
   else --extract 
      print("extracting a valid and a test subcorpora")
      if params.validp~=0 then
   	 local vdata = extract_data(datas[i], params.validp, params.valids, true)
   	 table.insert(vdatas, vdata)
      else
   	 table.insert(vdatas, tdata)
      end
      local tdata
      if not params.notestcorpus then
	 local data = extract_data(datas[i], params.validp, params.valids, true)
	 table.insert(tdatas, tdata)
      end
      local subtraindata = extract_data(datas[i], params.validp, params.valids, false)
      table.insert(subtraindatas, subtraindata)      
   end
end


-- printw(datas[1].words[1], datas[1].wordhash)
-- print(datas[1].words.sent[1])
-- printw(datas[1].words[2], datas[1].wordhash)
-- print(datas[1].words.sent[2])
-- printw(datas[1].words[3], datas[1].wordhash)
-- print(datas[1].words.sent[3])

-- printw(vdatas[1].words[1], vdatas[1].wordhash)
-- print(vdatas[1].words.sent[1])

-- printw(tdatas[1].words[1], tdatas[1].wordhash)
-- print(tdatas[1].words.sent[1])
--  io.read()


if #params.select_data~=0 then
   print(params.select_data)
   for i=1,#params.select_data do
      local f=torch.DiskFile(params.select_data[i])
      local t = f:readObject()
      print(#t)
      print(datas[i].size)
      assert(datas[i].size==#t)
      exit()
      --datas[i]
   end
end

local dataidx = {}
for i=1,#datas do
   if params.truesgd then
      for j=1,datas[i].size do
	 local n = datas[i].entities.nent(datas[i], j)
	 local nrel = ((n * (n-1))/2)
	 --print(n .. " " .. ((n * (n-1))/2))
	 for k=1, nrel do
	    table.insert(dataidx,{i,j,k})
	 end
      end
   else
      for j=1,datas[i].size do
	 table.insert(dataidx,{i,j})
      end
   end
end

params.nword = math.min(params.nword, #datas[1].wordhash)

print("creating network")
local network = createnetworks(params,datas)


if frestart then
   
   print('reloading network')
   if params.restart:match("model_net") then
      print("net")
      network = frestart:readObject()
   else
      print("weight")
      net = frestart:readObject()
      network:loadnet(params, net)
   end
   print("now testing")
   local d = params.corpus=="medline" and 'MedLine' or (params.corpus=='drugBank' and 'Drugbank' or 'Full')



   if true then
      params.brat=true
      local macro_p,macro_r,macro_f1,c,micro_p,micro_r,micro_f1 = test(network, vdatas[1], params)
      print("Valid_macro: " .. macro_p .. " " .. macro_r .. " " .. macro_f1)
      print("Valid_micro: " .. micro_p .. " " .. micro_r .. " " .. micro_f1)
      params.brat=false
      

      local ddata =  createdata(params, params.corpus[1], "train", params.decodedir)
      params.brat=true
      local macro_p,macro_r,macro_f1,c,micro_p,micro_r,micro_f1 = test(network, ddata, params)
      print("Valid_macro: " .. macro_p .. " " .. macro_r .. " " .. macro_f1)
      print("Valid_micro: " .. micro_p .. " " .. micro_r .. " " .. micro_f1)
      params.brat=false
      
      exit()
   end

   
   for i=1,#datas do
      params.brat=true
      local macro_p,macro_r,macro_f1,c,micro_p,micro_r,micro_f1 = test(network, vdatas[i], params)
      print("Valid_macro: " .. macro_p .. " " .. macro_r .. " " .. macro_f1)
      print("Valid_micro: " .. micro_p .. " " .. micro_r .. " " .. micro_f1)
      params.brat=false
      io.read()
      params.brat=true
      local macro_p,macro_r,macro_f1,c,micro_p,micro_r,micro_f1 = test(network, tdatas[i], params)
      print("Test_macro: " .. macro_p .. " " .. macro_r .. " " .. macro_f1)
      print("Test_micro: " .. micro_p .. " " .. micro_r .. " " .. micro_f1)
      params.brat=false
      --exit()
   end
   -- print(p)
   -- print(r)
   -- print(f1)
   exit()
end

local networksave = network:getnetsave(params)

local criterion = nn.ClassNLLCriterion()

local parameters_function
if params.lnormnobias then
   function parameters_function(self)
      if self.weight and self.bias then
	 return {self.weight}, {self.gradWeight}
      elseif self.weight then
	 return {self.weight}, {self.gradWeight}
      else
	 return
      end
   end
end


local myperm = torch.Tensor()
local tempperm = torch.Tensor()
function myrandperm(dataidx, datas)
   
   local sizes = torch.Tensor(#datas):fill(0)
   --computing sizes and min
   for i=1, #dataidx do
      sizes[ dataidx[i][1] ] = sizes[ dataidx[i][1] ] + 1
   end
   local min = sizes:min()

   print('sizes')
   print(sizes)
   
   myperm:resize(min * #datas):fill(0)
   
   local cumul=0
   for i=1,#datas do
      local r = torch.randperm(sizes[i]) + cumul
      myperm:narrow(1, ((i-1)*min) +1, min):copy( r:narrow(1,1,min) )
      cumul = cumul + sizes[i]
   end

   return myperm:index(1,torch.randperm(myperm:size(1)):long())
end


params.best = params.best or 0

--local nf = 0

print("now training")
local cost = 0
local ntoolong, nnoent, nforward, nsent = 0, 0, 0, 0

local fwddatas = torch.Tensor(#datas):fill(0)

local inputwords = torch.Tensor()
local inputentities = torch.Tensor()
local gradinput = torch.Tensor()
local targets = torch.Tensor()

local currentmax = 0

local iter = 0

while true do
   iter = iter + 1
   if iter>params.niter then exit() end
   if params.debug2 then print("---------------------------------------Training--------------------------------------------------") end
   
   local timer2
   local timeforward, timebackward, timegetgraph = 0, 0, 0
   if params.time then timer2 = torch.Timer() end
   local timer = torch.Timer()
   timer:reset()

   network:training()
   
   if params.dropout~=0 then
      --if type(network.dropout)~="table" then error("check that") end
      if params.dp==1 or params.dp==3 or params.dp==4 then
       	 network.dropout:training()
      end
      if params.dt==2 or params.dt==3 then
	 for i=1, #network.RNNs do
	    network.RNNs[i]:training()
	 end
      end
   end
   
   local perm
   if params.crossbalance==0 then
      perm = torch.randperm(#dataidx)
   else
      perm = myrandperm(dataidx, datas)
   end
   
   local nex = 0
   for i=1, perm:size(1) do
      local datacorpus = (not params.nosgd) and dataidx[perm[i]][1] or dataidx[i][1]
      local idx = (not params.nosgd) and dataidx[perm[i]][2] or dataidx[i][2]
      if (not params.nosgd) then fwddatas[ dataidx[perm[i]][1] ] = fwddatas[ dataidx[perm[i]][1] ] + 1 end
     -- print("====================== corpus " .. datacorpus .. " sentence " .. idx .. " size " .. datas[datacorpus].words[idx]:size(1) .. " nb entities " .. datas[datacorpus].entities.nent(datas[datacorpus], idx))
      --printw(datas[datacorpus].words[idx], datas[datacorpus].wordhash)
      
      --collectgarbage()
      if i%500==0 then
	 print(i .. " / " .. perm:size(1) .. "(" .. string.format('%.2f', nforward/timer:time().real) .. " ex/s)")
	 collectgarbage()
      end

      local words = datas[datacorpus].words[idx]
      if words:size(1)>params.maxsize or (params.maxent and datas[datacorpus].entities.nent(datas[datacorpus], idx)>params.maxent) then
	 ntoolong=ntoolong+1;
	 print("skip sentence " .. idx);
	 goto continue
      end
      
      if params.arch==3 and datas[datacorpus].entities.nent(datas[datacorpus], idx)>=2 then
	 if params.time then timer2:reset() end
	 network:getGraph(datas[datacorpus].trees[idx], datas[datacorpus])
	 if params.time then timegetgraph = timegetgraph + timer2:time().real end
      end
      
      --print(datas[datacorpus].entities.nent(datas[datacorpus], idx) .. " entities")
      if datas[datacorpus].entities.nent(datas[datacorpus], idx)<2 then
	 nnoent = nnoent + 1
      else
	 local n = datas[datacorpus].entities.nent(datas[datacorpus], idx)
	 nforward = nforward + ((n * (n-1))/2)
	 nsent = nsent + 1
      end


      if params.batch~=0 then
	 error("not anymore")
      else --no batch
	 if (params.arch==2 or params.dp==2 or params.dp==3 or params.rnn=="lstm" or params.rnn=="cnn") and (not (params.arch==4 or params.arch==5)) then words = words:view(1,words:size(1)) end

	 local nrel = 0
	 for ent1=1,datas[datacorpus].entities.nent(datas[datacorpus], idx) do
	    for ent2=ent1+1,datas[datacorpus].entities.nent(datas[datacorpus], idx) do
	       
	       --nf = nf +1
	       --print(nf)
	       --printw(words, datas[1].wordhash)
	       --print(datas[datacorpus].relations[idx])
	       
	       nrel = nrel + 1
	       local fwd_truesgd = true
	       if params.truesgd then
		  if nrel~=dataidx[perm[i]][3] then
		     fwd_truesgd=false;
		     --print("do not forward relation between " .. ent1 .. " and " .. ent2 .. " " .. idx) else print("do forward")
		  end
	       end
	       
	       --print(_select(params, datas[datacorpus], idx,ent1,ent2))
	       if fwd_truesgd and is_related(params, datas[datacorpus], idx, ent1, ent2) and _select(params, datas[datacorpus], idx,ent1,ent2) then
		  --print(" sentence " .. idx .. " relation between " .. ent1 .. " and " .. ent2 .. " (" .. datas[datacorpus].relations:isrelated(idx, ent1, ent2) .. ")")
		  local entities = datas[datacorpus].entities.getent(datas[datacorpus], idx, ent1, ent2)

		  --print(entities)
		  if (params.arch==2 or params.dp==2 or params.dp==3 or params.rnn=="lstm" or params.rnn=="cnn") and (not (params.arch==4 or params.arch==5)) then entities = entities:view(1, entities:size(1)) end
		  
		  local input = {words}

		  
		  if params.tfsz~=0 then table.insert(input, datas[datacorpus].entities.getenttags(datas[datacorpus], idx)) end
		  if params.pfsz~=0 then table.insert(input, datas[datacorpus].pos[idx]) end
		  if params.rdfsz~=0 then
		     table.insert(input, datas[datacorpus].get_relative_distance(entities, 1))
		     table.insert(input, datas[datacorpus].get_relative_distance(entities, 2))
		  end
		  if params.dtfsz~=0 and params.dt==1 then
		     local dtf = datas[datacorpus].deptypes[idx]
		     if (params.dp==2 or params.dp==3 or params.rnn=="lstm" or params.rnn=="cnn") and (not (params.arch==4 or params.arch==5)) then dtf = dtf:view(1, dtf:size(1)) end
		     table.insert(input, dtf)
		  end
		  table.insert(input, entities)
		  if params.dtfsz~=0 and params.dt==2 then
		     local dtf = datas[datacorpus].deptypes[idx]
		     if (params.dp==2 or params.dp==3 or params.rnn=="lstm" or params.rnn=="cnn") and (not (params.arch==4 or params.arch==5)) then dtf = dtf:view(1, dtf:size(1)) end
		     input = {input, dtf}
		  end

		  if params.arch==5 then
		     --print(entities)
		     datas[datacorpus].trees2[idx]:LCA(3,4,entities,1,2)
		     datas[datacorpus].trees2[idx]:set_tab_sontags(1, 2)
		     --datas[datacorpus].trees2[idx]:print_tree("")
		     --io.read()
		  end
		  
		  --debug
		  if params.debug then
		     nngraph.setDebug(true)
		     network.g.name = 'my_bad_linear_net'
		     pcall(function() network:forward(input) end)
		     os.execute('echo my_bad_linear_net.svg')
		  end

		  if params.time then timer2:reset() end


		  if params.aotf then
		     -- print('==========')
		     -- print(entities)
		     -- print(datas[datacorpus].words.sent[i])
		     -- printw(datas[datacorpus].words[i], datas[datacorpus].wordhash)
		     -- printw(words, datas[datacorpus].wordhash)
		     local new_input = {}
		     for inp=1,#input do
			local new_tab = {}
			local previous_tab = input[inp]
			local k = 1
			while k<=words:size(1) do
			   if entities[k]==2 or entities[k]==1 then --1=padding, 2=other
			      table.insert(new_tab, previous_tab[k])
			      k = k+1
			   else
			      --print(entities)
			      local ent = entities[k]
			      if inp==1 then --words
				 table.insert(new_tab, datas[datacorpus].wordhash["drug0"])
			      else --other features
				 table.insert(new_tab, previous_tab[k])
			      end
			      while entities[k]==ent do k = k + 1 end
			   end
			end
			table.insert(new_input, torch.Tensor(new_tab)) --memory allocation is bad
			-- print(previous_tab)
			-- print(new_tab)
			-- io.read()
		     end

		     input = new_input
		  end

		  local output
		  if params.arch==1 or params.arch==3 or params.arch==6 then
		     output = network:forward(input, datas[datacorpus].corpus)
		  elseif params.arch==4 or params.arch==5 then
		     -- print(datas[datacorpus].trees[idx])
		     -- datas[datacorpus].trees2[idx]:print_tree("")
		     -- io.read()
		     output = network:forward(datas[datacorpus].trees2[idx], input, datas[datacorpus].corpus)
		  else
		     error("")
		  end
		  if params.time then timeforward = timeforward + timer2:time().real end
		  
		  local target = datas[datacorpus].relations:isrelated(idx, ent1, ent2)
		  
		  --printw(input[1], datas[datacorpus].wordhash)
		  --print(datas[datacorpus].entities[idx])
		  --print(datas[datacorpus].relations[idx])
		  --print(target)
		  --io.read()

		  --print(network.network:get(2).output:sum())
		  if params.debug2 then printinput(words, datas[datacorpus].wordhash, input[2], datas[datacorpus].wordhash) end
		  
		  
		  if params.debug2 then
		     local max, indice = output:max(1)--caution: comment this!!!!!!!!!!!!!!
		     print("old " .. target .. " " .. indice[1])
		     for i=1,output:size(1) do io.write(output[i] .. " ") end; io.write("\n")
		  end

		  cost = cost + criterion:forward(output, target)
		  local grad = criterion:backward(output, target)

		  if (params.l1norm and params.l1norm~=0) or (params.l2norm and params.l2norm ~= 0) then
		     local backparameters_function
		     if params.lnormnobias then
			backparameters_function = nn.Linear.parameters 
			nn.Linear.parameters = parameters_function
			nn.TemporalConvolution.parameters = parameters_function
		     end
			
		     -- locals:
		     local norm,sign= torch.norm,torch.sign

		     local parameters, gradParameters
		     if params.lnormpos=='model' then
			parameters, gradParameters = network.treelstm:parameters()
		     elseif params.lnormpos=='scorer' then
			parameters, gradParameters = {}, {}
			for i=1,#datas do
			   local p,g = network.scorers[datas[i].corpus]:parameters()
			   for j=1,#p do
			      table.insert(parameters, p[j])
			      table.insert(gradParameters, g[j])
			   end
			end
		     elseif params.lnormpos=='emb' then
			parameters, gradParameters = network.lookup:parameters()
		     elseif params.lnormpos=='full' then
			parameters, gradParameters = network.treelstm:parameters()
			for i=1,#datas do
			   local p,g = network.scorers[datas[i].corpus]:parameters()
			   for j=1,#p do
			      table.insert(parameters, p[j])
			      table.insert(gradParameters, g[j])
			   end
			end
			local p,g = network.lookup:parameters()
			for i=1,#p do
			   table.insert(parameters, p[i])
			   table.insert(gradParameters, g[i])
			end
		     end
		     for i=1,#parameters do
			-- Loss:
			cost = cost + params.l1norm * norm(parameters[i],1)
			cost = cost + params.l2norm * norm(parameters[i],2)^2/2
		     
			-- Gradients:
			gradParameters[i]:add( sign(parameters[i]):mul(params.l1norm) + parameters[i]:clone():mul(params.l2norm) )
		     end
		     
		     if params.lnormnobias then
			nn.Linear.parameters = backparameters_function
			nn.TemporalConvolution.parameters = backparameters_function
		     end
		  end
		  
		  network:zeroGradParameters()
		  if params.time then timer2:reset() end
		  if params.arch==2 or params.arch==3 then
		     network:backward(input, grad)
		     network:updateParameters(params.lr)
		  elseif params.arch==4 or params.arch==5 then
		     network:backward(datas[datacorpus].trees2[idx], input, datas[datacorpus].corpus, grad)
		     --
		     local lr = params.balance and (params.lr / datas[datacorpus].relations.distribution[target]) or params.lr 
		     --print(target)
		     --print(toto .. " " .. target)
		     --print(params.lr)
		     network:updateParameters(params.balance and (params.lr / datas[datacorpus].relations.distribution[target]) or params.lr)
		     if lr==(1/0) then
			error("")
			--print(datas[datacorpus].relations[idx])x
			--io.read()
		     end
		  else
		     network:backwardUpdate(input, datas[datacorpus].corpus, grad, params.lr)
		  end
		  if params.time then timebackward = timebackward + timer2:time().real end


		  if params.debug2 then
		     if params.arch==1 or params.arch==3 then--caution comment this!!!!!!!!!
			output = network:forward(input, datas[datacorpus].corpus)
		     elseif params.arch==4 or params.arch==5 then
			output = network:forward(datas[datacorpus].trees2[idx], input, datas[datacorpus].corpus)
		     end
		     
		     local max, indice = output:max(1)--caution: comment this!!!!!!!!!!!!!!
		     print("new " .. target .. " " .. indice[1])
		     for i=1,output:size(1) do io.write(output[i] .. " ") end; io.write("\n") 
		  end

	       else
		  --print(nf .. " do not forward (can not be related)")
	       end
	       --io.read()
	    end
	 end
      end
      ::continue::
      
   end

   -- print(datas[1].relationhash)
   print(relation_pos)

   -- local total = 0
   -- for k,v in pairs(relation_pos) do
   --    total = total + v
   -- end
   -- print(total)
   -- exit()
   
   -- print(fwddatas)
   
   local file = io.open(rundir .. "/cost", 'a')
   print("cost " .. cost/nforward)
   local t = timer:time().real
   print(string.format('# ex/s = %.2f [%d ex over %d processed -- %.4g%%] %.2f s', nforward/t, nex, perm:size(1), nex/perm:size(1)*100, t))
   file:write(cost/nforward .. "\n")
   file:close()
   cost = 0
   --io.read()
   
   print(nsent .. " / " .. perm:size(1) .. " sentences forwarded (" .. nforward .. " possible relations)")
   print(ntoolong .. " long sentences skipped")
   print(nnoent .. " sentences with less than 2 entities")

   ntoolong, nnoent, nforward, nsent = 0, 0, 0, 0 


   if params.time then
      print("time forward : " .. timeforward)
      print("time backward : " .. timebackward)
      print("time getgraph : " .. timegetgraph)
      timeforward = 0
      timebackward = 0
      timegegraph = 0
      io.read()
   end

   print('saving: last model')
   local f = torch.DiskFile(string.format('%s/model.bin', rundir), 'w'):binary()
   f:writeObject(params)
   f:writeObject(networksave)
   f:close()

   -- local f = torch.DiskFile(string.format('%s/model_net.bin', rundir), 'w'):binary()
   -- f:writeObject(params)
   -- f:writeObject(network)
   -- f:close()   
   print("---------------------------------------Testing--------------------------------------------------")
   if not params.notest then
      print("============================================================================")
      print("================================now testing=================================")
      print("============================================================================")

      --test on valid (youpi!)
      local f1
      print("*****************************Test on valid***********************************")
      for i=1,#datas do
	 if params.testc==0 or params.testc==i then
	    local fcost = io.open(rundir .. "/" .. datas[i].corpus .. "-cost_valid", 'a')
	    local f_macro_precision = io.open(rundir .. "/" .. datas[i].corpus .. "-macro_precision_valid", 'a')
	    local f_macro_recall = io.open(rundir .. "/" .. datas[i].corpus .. "-macro_recall_valid", 'a')
	    local f_macro_f1 = io.open(rundir .. "/" .. datas[i].corpus .. "-macro_f1-score_valid", 'a')
	    local f_micro_precision = io.open(rundir .. "/" .. datas[i].corpus .. "-micro_precision_valid", 'a')
	    local f_micro_recall = io.open(rundir .. "/" .. datas[i].corpus .. "-micro_recall_valid", 'a')
	    local f_micro_f1 = io.open(rundir .. "/" .. datas[i].corpus .. "-micro_f1-score_valid", 'a')
	    
	    local macro_p,macro_r,macro_f1,c,micro_p,micro_r,micro_f1 = test(network, vdatas[i], params)
	    print("Valid_macro: " .. macro_p .. " " .. macro_r .. " " .. macro_f1)
	    print("Valid_micro: " .. micro_p .. " " .. micro_r .. " " .. micro_f1)
	    
	    f_macro_precision:write(macro_p .. "\n"); f_macro_precision:flush()
	    f_macro_recall:write(macro_r .. "\n"); f_macro_recall:flush()
	    f_macro_f1:write(macro_f1 .. "\n"); f_macro_f1:flush()
	    f_micro_precision:write(micro_p .. "\n"); f_micro_precision:flush()
	    f_micro_recall:write(micro_r .. "\n"); f_micro_recall:flush()
	    f_micro_f1:write(micro_f1 .. "\n"); f_micro_f1:flush()
	    fcost:write(c .. "\n"); fcost:flush()
	    
	    fcost:close()
	    f_macro_precision:close(); f_macro_recall:close(); f_macro_f1:close()
	    f_micro_precision:close(); f_micro_recall:close(); f_micro_f1:close()
	    f1 = macro_f1
	 end
      end

      if f1 > params.best then
      	 params.best = f1
      	 print('saving test: better than ever ' .. f1)
      	 local f = torch.DiskFile(string.format('%s/model-best-valid.bin', rundir), 'w'):binary()
      	 f:writeObject(params)
      	 f:writeObject(networksave)
      	 f:close()
      end

      print("*****************************Test on train***********************************")
      --test on train
      for i=1,#datas do
	 if params.testc==0 or params.testc==i then
	    local fcost = io.open(rundir .. "/" .. datas[i].corpus .. "-cost_train", 'a')
	    local f_macro_precision = io.open(rundir .. "/" .. datas[i].corpus .. "-macro_precision_train", 'a')
	    local f_macro_recall = io.open(rundir .. "/" .. datas[i].corpus .. "-macro_recall_train", 'a')
	    local f_macro_f1 = io.open(rundir .. "/" .. datas[i].corpus .. "-macro_f1-score_train", 'a')
	    local f_micro_precision = io.open(rundir .. "/" .. datas[i].corpus .. "-micro_precision_train", 'a')
	    local f_micro_recall = io.open(rundir .. "/" .. datas[i].corpus .. "-micro_recall_train", 'a')
	    local f_micro_f1 = io.open(rundir .. "/" .. datas[i].corpus .. "-micro_f1-score_train", 'a')
	    
	    local macro_p,macro_r,macro_f1,c,micro_p,micro_r,micro_f1 = test(network, subtraindatas[i], params)
	    print("Train_macro: " .. macro_p .. " " .. macro_r .. " " .. macro_f1)
	    print("Train_micro: " .. micro_p .. " " .. micro_r .. " " .. micro_f1)
	    
	    f_macro_precision:write(macro_p .. "\n"); f_macro_precision:flush()
	    f_macro_recall:write(macro_r .. "\n"); f_macro_recall:flush()
	    f_macro_f1:write(macro_f1 .. "\n"); f_macro_f1:flush()
	    f_micro_precision:write(micro_p .. "\n"); f_micro_precision:flush()
	    f_micro_recall:write(micro_r .. "\n"); f_micro_recall:flush()
	    f_micro_f1:write(micro_f1 .. "\n"); f_micro_f1:flush()
	    fcost:write(c .. "\n"); fcost:flush()
	    
	    fcost:close()
	    f_macro_precision:close(); f_macro_recall:close(); f_macro_f1:close()
	    f_micro_precision:close(); f_micro_recall:close(); f_micro_f1:close()
	    f1 = macro_f1
	 end
      end


      if not params.notestcorpus then
	 --test on test (bouhou!)
	 print("*****************************Test on test***********************************")
	 for i=1,#datas do
	    if params.testc==0 or params.testc==i then
	       local fcost = io.open(rundir .. "/" .. datas[i].corpus .. "-cost_test", 'a')
	       local f_macro_precision = io.open(rundir .. "/" .. datas[i].corpus .. "-macro_precision_test", 'a')
	       local f_macro_recall = io.open(rundir .. "/" .. datas[i].corpus .. "-macro_recall_test", 'a')
	       local f_macro_f1 = io.open(rundir .. "/" .. datas[i].corpus .. "-macro_f1-score_test", 'a')
	       local f_micro_precision = io.open(rundir .. "/" .. datas[i].corpus .. "-micro_precision_test", 'a')
	       local f_micro_recall = io.open(rundir .. "/" .. datas[i].corpus .. "-micro_recall_test", 'a')
	       local f_micro_f1 = io.open(rundir .. "/" .. datas[i].corpus .. "-micro_f1-score_test", 'a')
	       
	       local macro_p,macro_r,macro_f1,c,micro_p,micro_r,micro_f1 = test(network, tdatas[i], params)
	       print("Test_macro: " .. macro_p .. " " .. macro_r .. " " .. macro_f1)
	       print("Test_micro: " .. micro_p .. " " .. micro_r .. " " .. micro_f1)
	       
	       f_macro_precision:write(macro_p .. "\n"); f_macro_precision:flush()
	       f_macro_recall:write(macro_r .. "\n"); f_macro_recall:flush()
	       f_macro_f1:write(macro_f1 .. "\n"); f_macro_f1:flush()
	       f_micro_precision:write(micro_p .. "\n"); f_micro_precision:flush()
	       f_micro_recall:write(micro_r .. "\n"); f_micro_recall:flush()
	       f_micro_f1:write(micro_f1 .. "\n"); f_micro_f1:flush()
	       fcost:write(c .. "\n"); fcost:flush()
	       
	       fcost:close()
	       f_macro_precision:close(); f_macro_recall:close(); f_macro_f1:close()
	       f_micro_precision:close(); f_micro_recall:close(); f_micro_f1:close()
	    end
	 end
      end
   else
      for i=1,#datas do
	 if params.testc==0 or params.testc==i then
	    local macro_p,macro_r,macro_f1,c,micro_p,micro_r,micro_f1 = test(network, datas[i], params)
	    print("Train_macro: " .. macro_p .. " " .. macro_r .. " " .. macro_f1)
	    print("Train_micro: " .. micro_p .. " " .. micro_r .. " " .. micro_f1)
	 end
      end
   end
end
