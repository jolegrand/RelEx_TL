require 'torch'

local function tree2tree(trees)
   --print(string.format('tree2tree'))
   local newtrees = {}
   for i=1,#trees do
      local tree = trees[i]
      --print(tree)
      local reps = {}
      local j = 1
      while j<#tree do
	 local size = tree[j]
	 j = j+1
	 local head = treelstm.Tree()
	 head.idx = tree[j]
	 j = j + 1
	 for k=1,size-1 do
	    if tree[j]<1000 then
	       local son = treelstm.Tree()
	       son.idx = tree[j]
	       head:add_child(son)
	    else
	       head:add_child(reps[ tree[j]-1000 ])
	    end
	    j = j + 1
	 end
	 table.insert(reps, head)
      end
      for i=1,#reps do
	 --print(i)
	 --reps[i]:print()
      end
      -- print("==============")
      -- reps[#reps]:print()
      table.insert(newtrees, reps[#reps])
      --io.read()
   end
   return newtrees
end

local function loadtrees(filename, maxid)
   print(string.format('loading <%s>', filename))
   local trees = {}
   for line in io.lines(filename) do
      if line~="" then
	 if maxload and maxload > 0 and maxload == #trees then
	    print("breakdata10")
	    break
	 end
	 local tab = {}
	 for word in line:gmatch('(%d+)') do
	    table.insert(tab, tonumber(word))
	 end
	 table.insert(trees, tab)
      end
   end
   return trees
end

local function _loadhash(filename, maxidx)
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

local function _addhash(filename, hash)
   print(string.format('adding <%s> to hash', filename))
   local idx = #hash
   local _added, _present = 0, 0
   for key in io.lines(filename) do
      if not hash[key] then
	 _added = _added + 1
	 idx = idx + 1
	 table.insert(hash, key)
	 hash[key] = idx
      else
	 _present = _present + 1
      end
   end
   print(_added .. " words added, " .. _present .. " words already in hash")
   return hash
end

local function wordfeature(word)
   word = word:lower()
   word = word:gsub('%d+', '0')
   return word
end

local function loadindices(filename, maxload)
   print(string.format('loading <%s>', filename))
   local res = {}
   for line in io.lines(filename) do
      table.insert(res, line)
   end
   return res
end

local function loadwords(filename, hash, addraw, feature, maxload)
   print(string.format('loading <%s>', filename))
   local lines = addraw and {} or nil
   local indices = {}
   local sentences = {}
   for line in io.lines(filename) do
      local l = line:gsub(" +", " ")
      table.insert(sentences, l)
      if line~="" then
	 if maxload and maxload > 0 and maxload == #indices then
	    print("breakdata10")
	    break
	 end
	 local words = {}
	 local wordsidx = {}
	 for word in line:gmatch('(%S+)') do
	    if addraw then
	       table.insert(words, word)
	    end
	    table.insert(wordsidx, hash[feature and feature(word) or word] or hash.UNK)
	 end
	 if addraw then
	    table.insert(lines, words)
	 end
	 
	 table.insert(indices, torch.Tensor(wordsidx))
      end
   end

   --print("nb line " .. #indices)

   --print(lines)
   
   collectgarbage()
   return {raw=lines, idx=indices, sent=sentences}
end

local function idx(tbl)
   setmetatable(tbl, {__index = function(self, idx)
                                   return self.idx[idx]
                                end})
end

local function pad(tbl, sz, val)
   setmetatable(tbl, {__index = function(self, idx)
			 local x = self.idx[idx]
			 local px = torch.Tensor(x:size(1)+2*sz):fill(val)
			 px:narrow(1, sz+1, x:size(1)):copy(x)
			 return px
                                end})
end


local function loadentities(filename, sents, words, hash, maxload, wsz)
   print(string.format('loading <%s>', filename))
      
   local entities = {}
   local count=0
   --local countprint = 267
   for line in io.lines(filename) do
      --print(count .. " " .. line)
      count = count + 1
      entities[count] = {}
      --print(#entities+1)
      --print(sents[#entities+1])
      --print(line)
      if line~="" then
	 if countprint and count==countprint then print(line); print(sents[count]); io.read() end
	 if maxload and maxload > 0 and #entities>maxload  then
	    print("break entities")
	    break
	 else
	    entities[count] = {}
	    for entity in line:gmatch("[^ %d]+ [%d ]+") do
	       --load boundaries
	       local _type = entity:match("([^ ]+)")
	       local bounds = {}
	       --print(_type)
	       for bound in entity:gmatch("%d+ %d+") do
		  local i1 = bound:match("(%d+) %d+")
		  local i2 = bound:match("%d+ (%d+)")
		  i1 = tonumber(i1)
		  i2 = tonumber(i2)
		  --print(i1)
		  --print(i2)
		  table.insert(bounds, {i1,i2})
	       end
	       --print(bounds)
	       --print(sents[count])

	       if countprint and count==countprint then print(bounds); io.read()end

	       --get the corresponding words
	       local words = {}
	       local enti=1
	       local id = 0
	       local idw = 1
	       local boolent = false
	       if id>=bounds[enti][1] then --sentence start with a drug
		  table.insert(words, idw)
		  boolent = true
		  --print(idw)
	       end
	       --print(#sents)
	       for i=1,#sents[count] do
		  if countprint and count==countprint then print(sents[count]:sub(i,i) .. " " .. i .. " " .. id) end
		  if sents[count]:sub(i,i)~=" " then
		     --if countprint and count==countprint then print("hop") end
		     if id>=bounds[enti][2] then
			if countprint and count==countprint then print("end") end
			boolent = false
			enti=enti+1
			if not bounds[enti] then break end --entity entirely found
		     end
		     id = id + 1 
		  else
		     idw = idw + 1
		     if boolent==false and id>bounds[enti][1] then--start entity when it start in the middle of a word (the cutted entity is included)
			if countprint and count==countprint then print("anomaly") end
			table.insert(words, idw-1)--print(idw);
			boolent=true
		     end
		     if id==bounds[enti][1] then
			if countprint and count==countprint then print("start") end
			--table.insert(words, idw)--print(idw);
			boolent=true
		     end
		     if boolent then table.insert(words, idw) end--print(idw) end
		     --print("====" .. id .. " " .. bounds[enti][2])
		  end
	       end
	       if countprint and count==countprint then print("="); print(words);print("="); io.read() end
	       --if #words==0 then io.read() end
	       table.insert(entities[count], {_type, torch.IntTensor(words)})
	       --print(words)
	       --if #bounds>1 then io.read() end
	       --table.insert(entities[count], {_type, bounds})
	    end
	 end
      end
      --print(entities[count])
   end
   
   local pad = (wsz-1)/2
   local res = torch.Tensor()
   entities.getent = function(data, nsent, e1, e2)
      res:resize(data.words[nsent]:size(1)):fill(2)--2=Other
      for i=1,pad do
	 res[i]=1 --1=Padding
	 res[res:size(1)-i+1] = 1 --1=Padding
      end
      --printw(data.words[nsent], data.wordhash)
      --print(nsent)
      --print(data.words[nsent])
      --print(data.entities[nsent])
      local ent1 = data.entities[nsent][e1][2]
      local ent2 = data.entities[nsent][e2][2]
      for i=1,ent1:size(1) do res[ ent1[i] + pad ]=3 end--entity1
      for i=1,ent2:size(1) do res[ ent2[i] + pad ]=4 end--entity2
      return res
   end
   
   local res2 = torch.Tensor()
   entities.getenttags = function(data, nsent)
      res2:resize(data.words[nsent]:size(1) + (2*pad)):fill(hash["O"])--create input tensor
      for i=1,pad do
	 res2[i]=hash["PADDING"] --1=Padding
	 res2[res2:size(1)-i+1] = hash["PADDING"] --1=Padding
      end
      for i=1,#data.entities[nsent] do
	 local _type = data.entities[nsent][i][1]
	 for j=1,data.entities[nsent][i][2]:size(1) do
	    res2[data.entities[nsent][i][2][j] + pad] = hash[_type]
	 end
      end
      return res2
   end

   entities.nent = function(data, nsent)
      return #data.entities[nsent]
   end

   entities.typeent = function(data, nsent, nent)
      return data.entities[nsent][nent][1]
   end
   
   return entities
   
end

function loadrelations(filename, hash, maxload, params)
   print(string.format('loading <%s>', filename))
   local distribution = torch.Tensor(#hash):fill(0)
   local relations = {}
   local count = 0
   for line in io.lines(filename) do
      count = count + 1
      relations[count] = {}
      --print(#entities+1)
      --print(sents[#entities+1])
      if not line:match("^[\t ]*$") then
	 if maxload and maxload > 0 and #relations>maxload then
	    print("break relations")
	    break
	 else
	    --print(line)
	    for relation in line:gmatch("%d+ %d+ [^%d ]+") do
	       --print(relation)
	       local e1 = relation:match("%d+")
	       local e2 = relation:match("%d+ (%d+)")
	       local _type = relation:match("%d+ %d+ ([^%d ]+)")
	       e1 = tonumber(e1)+1
	       e2 = tonumber(e2)+1
	       --print(e1)
	       --print(e2)
	       if relations[count][e1]==nil then relations[count][e1]={} end
	       relations[count][e1][e2] = hash[_type]--, {_type, e2})
	       distribution[hash[_type]] = distribution[hash[_type]] + 1
	    end
	 end
      end
   end
 
   relations.isrelated = function(self, nsent, e1, e2)
      --print(self[nsent])
      --print(nsent)
      --print(e1)
      --print(e2)
      return self[nsent][e1] and self[nsent][e1][e2] or hash["null"] 
   end

   -- for i=1,distribution:size(1) do
   --    distribution[i] = distribution[i]==0 and 0 or ( 1 / distribution[i])
   -- end
   --distribution:norm()
   --distribution = distribution / distribution:max()
   --print(distribution)
   local min = math.huge
   for i=1,distribution:size(1) do
      if distribution[i]~=0 and distribution[i]<min then min = distribution[i] end
   end
   distribution = distribution / min
   --print(distribution)
   relations.distribution = distribution
   
   return relations
end
   
local wordhash, enthash, deptypehash, poshash

function loadhash(params, corpus)
   --local relationhash
   
   if corpus=="medline" or corpus=="drugbank" or corpus=="full" then
      local path, pathdata, file, corpusback
      if params.mobius then
	 local home = "/home/runuser"
	 path = home .. "/corpus/semevalData/"
      else
	 path = "/home/joel/Bureau/loria/corpus/Semeval2013DDI/"
      end
      wordhash = wordhash
	 and _addhash(path .. 'hash' .. (params.anon and "_anon" or "") .. '/word.txt', wordhash)
	 or _loadhash(path .. 'hash' .. (params.anon and "_anon" or "") .. '/word.txt', params.nword)
      entityhash = entityhash
	 and _addhash(path .. 'hash' .. (params.anon and "_anon" or "") .. '/entities.txt', entityhash)
	 or _loadhash(path .. 'hash' .. (params.anon and "_anon" or "") .. '/entities.txt', nil)
      --relationhash = relationhash
      --and _addhash(path .. 'hash' .. (params.anon and "_anon" or "") .. '/relations.txt', relationhash)
      -- or _loadhash(path .. 'hash' .. (params.anon and "_anon" or "") .. '/relations.txt', nil)
      if params.pfsz~=0 then
	 poshash = poshash
	    and _addhash(path .. 'hash' .. (params.anon and "_anon" or "") .. '/pos.txt', poshash)
	    or _loadhash(path .. 'hash' .. (params.anon and "_anon" or "") .. '/pos.txt', nil)
      end
      if (params.arch==3 or params.arch==4 or params.arch==5) and params.dtfsz>0 then
	 deptypehash = deptypehash
	    and _addhash(path .. 'hash' .. (params.anon and "_anon" or "" ) .. '/deptype.txt', deptypehash)
	    or _loadhash(path .. 'hash' .. (params.anon and "_anon" or "" ) .. '/deptype.txt', nil)
      end
   elseif corpus=="snpphena" then
      local path, pathdata
      local tot = trainortest=="test" and "Test" or "Train"
      if params.mobius then
	 local home = "/home/runuser"
	 path = home .. "/corpus/SNPPhenA/SNPPhenA_BRAT/"
      else
	 path = "/home/joel/Bureau/loria/corpus/SNPPhenA/SNPPhenA_BRAT/"
      end
      local anon = false
      wordhash = wordhash
	 and _addhash(path .. 'hash' .. (anon and "_anon" or "") .. '/word.txt', wordhash)
	 or _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/word.txt', params.nword)
      entityhash = entityhash
	 and _addhash(path .. 'hash' .. (anon and "_anon" or "") .. '/entities.txt', entityhash)
	 or _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/entities.txt', nil)
      --relationhash = relationhash
      --and _addhash(path .. 'hash' .. (anon and "_anon" or "") .. '/relations.txt', relationhash)
      --or _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/relations.txt', nil)
      poshash = poshash
	 and _addhash(path .. 'hash' .. (anon and "_anon" or "") .. '/pos.txt', poshash)
	 or _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/pos.txt', nil)
      
      -- local weak = relationhash["weak_confidence_association"]
      -- relationhash["strong_confidence_association"] = weak
      -- relationhash["moderate_confidence_association"] = weak
      if (params.arch==3 or params.arch==4 or params.arch==5) and params.dtfsz>0 then
	 deptypehash = deptypehash
	    and _addhash(path .. 'hash' .. (anon and "_anon" or "") .. '/deptype.txt', deptypehash)
	    or _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/deptype.txt', nil)
	 --print(deptypehash)
      end
   elseif corpus=="PGxCorpus" then
      local path, pathdata
      local tot =  "Train"
      if params.mobius then
	 local home = "/home/runuser"
	 path = home .. "/corpus/PGxCorpus/"
      else
	 path = "/home/joel/Bureau/loria/corpus/PGxCorpus/"
      end
      local anon = false
      wordhash = wordhash
	 and _addhash(path .. 'hash' .. (anon and "_anon" or "") .. '/word.txt', wordhash)
	 or _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/word.txt', params.nword)
      entityhash = entityhash
	 and _addhash(path .. 'hash' .. (anon and "_anon" or "") .. '/entities.txt', entityhash)
	 or _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/entities.txt', nil)
      --relationhash = relationhash
      --and _addhash(path .. 'hash' .. (anon and "_anon" or "") .. '/relations.txt', relationhash)
      --or _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/relations.txt', nil)
      poshash = poshash
	 and _addhash(path .. 'hash' .. (anon and "_anon" or "") .. '/pos.txt', poshash)
	 or _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/pos.txt', nil)
      
      -- local weak = relationhash["weak_confidence_association"]
      -- relationhash["strong_confidence_association"] = weak
      -- relationhash["moderate_confidence_association"] = weak
      if (params.arch==3 or params.arch==4 or params.arch==5) and params.dtfsz>0 then
	 deptypehash = deptypehash
	    and _addhash(path .. 'hash' .. (anon and "_anon" or "") .. '/deptype.txt', deptypehash)
	    or _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/deptype.txt', nil)
	 --print(deptypehash)
      end
   elseif corpus=="reACE" then
      local path, pathdata
      local tot = trainortest=="test" and "Test" or "Train"
      if params.mobius then
	 local home = "/home/runuser"
	 path = home .. "/corpus/reACE/"
      else
	 path = "/home/joel/Bureau/loria/corpus/reACE/"
      end
      local anon = false
      wordhash = wordhash
	 and _addhash(path .. 'hash/word.txt', wordhash)
	 or _loadhash(path .. 'hash/word.txt', params.nword)
      entityhash = entityhash
	 and _addhash(path .. 'hash/entities.txt', entityhash)
	 or _loadhash(path .. 'hash/entities.txt', nil)
      --relationhash = relationhash
      --and _addhash(path .. 'hash/relations.txt', relationhash)
      --or _loadhash(path .. 'hash/relations.txt', nil)
      poshash = poshash
	 and _addhash(path .. 'hash/pos.txt', poshash)
	 or _loadhash(path .. 'hash/pos.txt', nil)
      
      if (params.arch==3 or params.arch==4 or params.arch==5) and params.dtfsz>0 then
	 deptypehash = deptypehash
	    and _addhash(path .. 'hash/deptype.txt', deptypehash)
	    or _loadhash(path .. 'hash/deptype.txt', nil)
      end
   elseif corpus=="ppi" or corpus=="LLL" or corpus=="AIMed" or corpus=="HPRD50" or corpus=="IEPA" or corpus=="BioInfer" then
      local path, pathdata
      local tot = trainortest=="test" and "Test" or "Train"
      if params.mobius then
	 local home = "/home/runuser"
	 path = home .. "/corpus/ppi-eval-standard/"
      else
	 path = "/home/joel/Bureau/loria/corpus/ppi-eval-standard/"
      end
      local anon = false
      wordhash = wordhash
	 and _addhash(path .. 'hash' .. (anon and "_anon" or "") .. '/word.txt', wordhash)
	 or _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/word.txt', params.nword)
      entityhash = entityhash
	 and _addhash(path .. 'hash' .. (anon and "_anon" or "") .. '/entities.txt', entityhash)
	 or _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/entities.txt', nil)
      --relationhash = relationhash
      --and _addhash(path .. 'hash' .. (anon and "_anon" or "") .. '/relations.txt', relationhash)
	 --or _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/relations.txt', nil)
      poshash = poshash
	 and _addhash(path .. 'hash' .. (anon and "_anon" or "") .. '/pos.txt', poshash)
	 or _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/pos.txt', nil)
      
      if (params.arch==3 or params.arch==4 or params.arch==5) and params.dtfsz>0 then
	 deptypehash = deptypehash
	    and _addhash(path .. 'hash/deptype.txt', deptypehash)
	    or _loadhash(path .. 'hash/deptype.txt', nil)
      end

   elseif corpus=="ADE" then
      local path, pathdata
      local tot = trainortest=="test" and "Test" or "Train"
      if params.mobius then
	 local home = "/home/runuser"
	 path = home .. "/corpus/" .. corpus .. "/"
      else
	 path = "/home/joel/Bureau/loria/corpus/" .. corpus .. "/"
      end
      local anon = false
      wordhash = wordhash
	 and _addhash(path .. 'hash' .. (anon and "_anon" or "") .. '/word.txt', wordhash)
	 or _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/word.txt', params.nword)
      entityhash = entityhash
	 and _addhash(path .. 'hash' .. (anon and "_anon" or "") .. '/entities.txt', entityhash)
	 or _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/entities.txt', nil)
      --relationhash = relationhash
	 --and _addhash(path .. 'hash' .. (anon and "_anon" or "") .. '/relations.txt', relationhash)
	 --or _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/relations.txt', nil)
      poshash = poshash
	 and _addhash(path .. 'hash' .. (anon and "_anon" or "") .. '/pos.txt', poshash)
	 or _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/pos.txt', nil)
      
      if (params.arch==3 or params.arch==4 or params.arch==5) and params.dtfsz>0 then
	 deptypehash = deptypehash
	    and _addhash(path .. 'hash' .. (anon and "_anon" or "") .. '/deptype.txt', deptypehash)
	    or _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/deptype.txt', nil)
      end
      
   elseif corpus=="EUADR_drug_disease" or corpus=="EUADR_drug_target" or corpus=="EUADR_target_disease" then
      local path, pathdata
      local tot = trainortest=="test" and "Test" or "Train"
      if params.mobius then
	 local home = "/home/runuser"
	 path = home .. "/corpus/EUADR/"
      else
	 path = "/home/joel/Bureau/loria/corpus/EUADR/"
      end
      local anon = false
      wordhash = wordhash
	 and _addhash(path .. 'hash' .. (anon and "_anon" or "") .. '/word.txt', wordhash)
	 or _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/word.txt', params.nword)
      entityhash = entityhash
	 and _addhash(path .. 'hash' .. (anon and "_anon" or "") .. '/entities.txt', entityhash)
	 or _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/entities.txt', nil)
      --relationhash = relationhash
	 --and _addhash(path .. 'hash' .. (anon and "_anon" or "") .. '/relations.txt', relationhash)
	 --or _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/relations.txt', nil)
      poshash = poshash
	 and _addhash(path .. 'hash' .. (anon and "_anon" or "") .. '/pos.txt', poshash)
	 or _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/pos.txt', nil)
      
      if (params.arch==3 or params.arch==4 or params.arch==5) and params.dtfsz>0 then
	 deptypehash = deptypehash
	    and _addhash(path .. 'hash' .. (anon and "_anon" or "") .. '/deptype.txt', deptypehash)
	    or _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/deptype.txt', nil)
      end
   else
      error("unknown corpus")
   end
   --print(#wordhash)
end

function createdata(params, corpus, trainortest, hashes)
   local fusr = {}
   print(params.fusr)
   for i=1,#params.fusr do
      fusr[params.fusr[i]]=i
   end
   
   local relationhash
   local parser = (not params.parser or params.parser=="stanford") and "corenlp.conll" or "McClosky.trees.ddg"
   
   if corpus=="medline" or corpus=="drugbank" or corpus=="full" then
      local path, pathdata, file, corpusback
      if params.mobius then
	 local home = "/home/runuser"
	 path = home .. "/corpus/semevalData/"
      else
	 path = "/home/joel/Bureau/loria/corpus/Semeval2013DDI/"
      end
      
      if trainortest=="train" or trainortest=="subtrain" then
	 if trainortest=="subtrain" then corpusback=corpus; corpus="subtrain" end
	 pathdata = path .. "Train/"
	 if params.anon then
	    file = pathdata .. "anon/" .. corpus .. parser .. "_tree_sentences"
	 else
	    file = pathdata .. "" .. corpus .. "sentences_tokens.txt"
	 end
      elseif trainortest=="test" then
	 pathdata = path .. "Test/DDI_Extraction/"
	 if params.anon then
	    file = pathdata .. "anon/" .. corpus .. "" .. parser .. "_tree_sentences"	 
	 else
	    file = pathdata .. "" .. corpus .. "sentences_tokens.txt"
	 end
      end
      
      -- wordhash = wordhash or _loadhash(path .. 'hash' .. (params.anon and "_anon" or "") .. '/word.txt', params.nword)
      -- entityhash = entityhash or _loadhash(path .. 'hash' .. (params.anon and "_anon" or "") .. '/entities.txt', nil)
      relationhash = _loadhash(path .. 'hash' .. (params.anon and "_anon" or "") .. '/relations.txt', nil)
      if fusr["full"] then
	 local int = relationhash["int"]
	 relationhash["effect"] = int
	 relationhash["mechanism"] = int
	 relationhash["advise"] = int
      end
      
      -- poshash = poshash or _loadhash(path .. 'hash' .. (params.anon and "_anon" or "") .. '/pos.txt', nil)
      -- if params.arch==3 and params.dtfsz>0 then
      -- 	 deptypehash = _loadhash(path .. 'hash' .. (params.anon and "_anon" or "" ) .. '/deptype.txt', nil)
      -- end
      
      local ids = loadindices(pathdata .. "raw/" .. corpus .. "sentenceids.txt", params.maxload)
      
      local words  = loadwords(file, wordhash, params.addraw, wordfeature, params.maxload)
      local entities = loadentities(pathdata .. (params.anon and "anon/" or "") .. corpus .. "entities.txt", words.sent, words.idx, entityhash, params.maxload, params.wsz)
      pad(words, (params.wsz-1)/2, wordhash.PADDING)
      local relations = loadrelations(pathdata .. "raw/" .. corpus .. "relations.txt", relationhash, params.maxload, params.wsz)

      
      --local trees = loadtrees(pathdata .. "" .. corpus .. "stanfordUD_tree_comp_reps", params.maxload)
      local trees = loadtrees(pathdata .. (params.anon and "anon/" or "") .. corpus .. "" .. parser .. "_tree_comp_reps", params.maxload)
      local trees2
      if params.arch==4 or params.arch==5 then
	 trees2 = tree2tree(trees)
      end

      local deptypes
      if (params.arch==3 or params.arch==4 or params.arch==5) and params.dtfsz>0 then
	 deptypes = loadwords(pathdata .. (params.anon and "anon/" or "") .. corpus .. "" .. parser .. "_tree_deptype", deptypehash, nil, nil, params.maxload)
	 idx(deptypes)
      end

      local pos
      if params.pfsz~=0 then
	 pos = loadwords(pathdata .. (params.anon and "anon/" or "") .. corpus .. "" .. parser .. "_tree_pos" , poshash, params.addraw, nil, params.maxload)
	 pad(pos, (params.wsz-1)/2, poshash.PADDING)
      end


      -- local get_relative_distance
      -- do 
      -- 	 local reldists = {torch.Tensor(), torch.Tensor()}
      -- 	 function get_relative_distance(ent, nent)
      -- 	    reldists[nent]:resizeAs(ent):fill(0)
      -- 	    local _, currentent = ent:eq(nent+2):max(1) --+2 for padding and nul
      -- 	    currentent = currentent[1]
      -- 	    --print(ent)
      -- 	    --print(currentent)
      -- 	    for i=1,reldists[nent]:size(1) do
      -- 	       reldists[nent][i]=math.abs(currentent-i)+1
      -- 	    end
      -- 	    return reldists[nent]
      -- 	 end
      -- end

      --local en = torch.Tensor({2,2,2,2,2,3,2,3,2,2,2,4,2})
      --print(get_relative_distance(en, 4))
      --exit()
      
      if trainortest=="subtrain" then corpus = corpusback end

      local select_data
      if trainortest=="train" and params.select_data and params.select_data[corpus] then
	 local f=torch.DiskFile(params.select_data[corpus])
	 select_data = f:readObject()
	 f:close()
	 assert(#words.idx==#select_data)
      end

      
      return {select_data=select_data, corpus=corpus, trees2=trees2, pos=pos, deptypes=deptypes, trees=trees, words=words, poshash=poshash, wordhash=wordhash, deptypehash=deptypehash, entityhash=entityhash, relationhash=relationhash, entities=entities, relations=relations, size=#words.idx, ids=ids, get_relative_distance=get_relative_distance}

   elseif corpus=="snpphena" then
      local path, pathdata
      local tot = trainortest=="test" and "Test" or "Train"
      if params.mobius then
	 local home = "/home/runuser"
	    path = home .. "/corpus/SNPPhenA/SNPPhenA_BRAT/"
      else
      	 path = "/home/joel/Bureau/loria/corpus/SNPPhenA/SNPPhenA_BRAT/"
      end
      pathdata = path .. "extracted/" ..  (copus=="train" and "Train" or "Test")

      local anon = false
      -- wordhash = wordhash or _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/word.txt', params.nword)
      -- entityhash = entityhash or _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/entities.txt', nil)
      relationhash = _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/relations.txt', nil)
      local weak = relationhash["weak_confidence_association"]
      relationhash["strong_confidence_association"] = weak
      relationhash["moderate_confidence_association"] = weak

      
      -- poshash = poshash or _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/pos.txt', nil)
      --local ids = loadindices(pathdata .. "raw/" .. corpus .. "sentenceids.txt", params.maxload)
      
      local words  = loadwords(path .. "extracted/" .. tot .. "/sentences." .. parser .. "_tree_sentences" , wordhash, params.addraw, wordfeature, params.maxload)
      local entities = loadentities(path .. "extracted/" .. tot .. "/entities.txt", words.sent, words.idx, entityhash, params.maxload, params.wsz)
      pad(words, (params.wsz-1)/2, wordhash.PADDING)
      local relations = loadrelations(path .. "extracted/" .. tot .. "/relations.txt", relationhash, params.maxload, params.wsz)
      local trees = loadtrees(path .. "extracted/" .. tot .. "/sentences." .. parser .. "_tree_comp_reps", params.maxload)
      local trees2
      if params.arch==4 or params.arch==5 then
	 trees2 = tree2tree(trees)
      end

      local deptypes
      if (params.arch==3 or params.arch==4 or params.arch==5) and params.dtfsz>0 then
	 deptypes = loadwords(path .. "extracted/" .. tot .. "/sentences." .. parser .. "_tree_deptype", deptypehash, nil, nil, params.maxload)
	 idx(deptypes)
	 -- 	 deptypehash = _loadhash(path .. '/deptype.txt', nil)
      end

      local pos
      if params.pfsz~=0 then
	 pos = loadwords(path .. "extracted/" .. tot .. "/sentences." .. parser .. "_tree_pos" , poshash, params.addraw, nil, params.maxload)
	 pad(pos, (params.wsz-1)/2, poshash.PADDING)
      end

      -- print(corpus)
      -- print(tree2)
      -- print(pos)
      -- print(deptypes)
      -- --print(trees) --ok
      -- --print(words) --ok
      -- --print(entities) --ok
      -- --print(relations) --ok
      -- --print(ids)
      -- exit()

      local select_data
      if trainortest=="train" and params.select_data and params.select_data[corpus] then
	 local f=torch.DiskFile(params.select_data[corpus])
	 select_data = f:readObject()
	 f:close()
	 assert(#words.idx==#select_data)
      end

     
      return {select_data=select_data, corpus=corpus, trees2=trees2, pos=pos, deptypes=deptypes, trees=trees, words=words, poshash=poshash, wordhash=wordhash, deptypehash=deptypehash, entityhash=entityhash, relationhash=relationhash, entities=entities, relations=relations, size=#words.idx, ids=ids, get_relative_distance=get_relative_distance}


   elseif corpus=="PGxCorpus" then
      local path, pathdata
      local tot = "Train"
      if params.mobius then
	 local home = "/home/runuser"
	 path = home .. "/corpus/PGxCorpus/"
      else
      	 path = "/home/joel/Bureau/loria/corpus/PGxCorpus/"
      end
      pathdata = path .. "extracted/full/" 

      local anon = false
      -- wordhash = wordhash or _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/word.txt', params.nword)
      -- entityhash = entityhash or _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/entities.txt', nil)
      relationhash = _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/relations.txt', nil)
      local weak = relationhash["weak_confidence_association"]
      relationhash["strong_confidence_association"] = weak
      relationhash["moderate_confidence_association"] = weak

      
      -- poshash = poshash or _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/pos.txt', nil)
      --local ids = loadindices(pathdata .. "raw/" .. corpus .. "sentenceids.txt", params.maxload)
      
      local words  = loadwords(pathdata .. "/sentences." .. parser .. "_tree_sentences" , wordhash, params.addraw, wordfeature, params.maxload)
      local entities = loadentities(pathdata .. "/entities.txt", words.sent, words.idx, entityhash, params.maxload, params.wsz)
      pad(words, (params.wsz-1)/2, wordhash.PADDING)
      local relations = loadrelations(pathdata .. "/relations.txt", relationhash, params.maxload, params.wsz)
      local trees = loadtrees(pathdata .. "/sentences." .. parser .. "_tree_comp_reps", params.maxload)
      local trees2
      if params.arch==4 or params.arch==5 then
	 trees2 = tree2tree(trees)
      end

      local deptypes
      if (params.arch==3 or params.arch==4 or params.arch==5) and params.dtfsz>0 then
	 deptypes = loadwords(path .. "extracted/" .. tot .. "/sentences." .. parser .. "_tree_deptype", deptypehash, nil, nil, params.maxload)
	 idx(deptypes)
	 -- 	 deptypehash = _loadhash(path .. '/deptype.txt', nil)
      end

      local pos
      if params.pfsz~=0 then
	 pos = loadwords(path .. "extracted/" .. tot .. "/sentences." .. parser .. "_tree_pos" , poshash, params.addraw, nil, params.maxload)
	 pad(pos, (params.wsz-1)/2, poshash.PADDING)
      end

      -- print(corpus)
      -- print(tree2)
      -- print(pos)
      -- print(deptypes)
      -- --print(trees) --ok
      -- --print(words) --ok
      -- --print(entities) --ok
      -- --print(relations) --ok
      -- --print(ids)
      -- exit()

      local select_data
      if trainortest=="train" and params.select_data and params.select_data[corpus] then
	 local f=torch.DiskFile(params.select_data[corpus])
	 select_data = f:readObject()
	 f:close()
	 assert(#words.idx==#select_data)
      end

     
      return {select_data=select_data, corpus=corpus, trees2=trees2, pos=pos, deptypes=deptypes, trees=trees, words=words, poshash=poshash, wordhash=wordhash, deptypehash=deptypehash, entityhash=entityhash, relationhash=relationhash, entities=entities, relations=relations, size=#words.idx, ids=ids, get_relative_distance=get_relative_distance}


      
   elseif corpus=="reACE" then
      local path, pathdata
      local tot = ""
      if params.mobius then
	 local home = "/home/runuser"
	    path = home .. "/corpus/reACE/"
      else
      	 path = "/home/joel/Bureau/loria/corpus/reACE/"
      end
      pathdata = path .. "extracted/"

      local anon = false
      -- wordhash = wordhash or _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/word.txt', params.nword)
      -- entityhash = entityhash or _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/entities.txt', nil)
      relationhash = _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/relations.txt', nil)
      local weak = relationhash["weak_confidence_association"]
      relationhash["strong_confidence_association"] = weak
      relationhash["moderate_confidence_association"] = weak

      
      -- poshash = poshash or _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/pos.txt', nil)
      --local ids = loadindices(pathdata .. "raw/" .. corpus .. "sentenceids.txt", params.maxload)
      
      local words  = loadwords(path .. "extracted/" .. tot .. "/sentences." .. parser .. "_tree_sentences" , wordhash, params.addraw, wordfeature, params.maxload)
      local entities = loadentities(path .. "extracted/" .. tot .. "/entities.txt", words.sent, words.idx, entityhash, params.maxload, params.wsz)
      pad(words, (params.wsz-1)/2, wordhash.PADDING)
      local relations = loadrelations(path .. "extracted/" .. tot .. "/relations.txt", relationhash, params.maxload, params.wsz)
      local trees = loadtrees(path .. "extracted/" .. tot .. "/sentences." .. parser .. "_tree_comp_reps", params.maxload)
      local trees2
      if params.arch==4 or params.arch==5 then
	 trees2 = tree2tree(trees)
      end
      local deptypes
      if (params.arch==3 or params.arch==4 or params.arch==5) and params.dtfsz>0 then
	 deptypes = loadwords(path .. "extracted/" .. tot .. "/sentences." .. parser .. "_tree_deptype", deptypehash, nil, nil, params.maxload)
	 idx(deptypes)
	 -- 	 deptypehash = _loadhash(path .. '/deptype.txt', nil)
      end
      local pos
      if params.pfsz~=0 then
	 pos = loadwords(path .. "extracted/" .. tot .. "/sentences." .. parser .. "_tree_pos" , poshash, params.addraw, nil, params.maxload)
	 pad(pos, (params.wsz-1)/2, poshash.PADDING)
      end


      local select_data
      if trainortest=="train" and params.select_data and params.select_data[corpus] then
	 local f=torch.DiskFile(params.select_data[corpus])
	 select_data = f:readObject()
	 f:close()
	 assert(#words.idx==#select_data)
      end

      
      return {select_data=select_data, corpus=corpus, trees2=trees2, pos=pos, deptypes=deptypes, trees=trees, words=words, poshash=poshash, wordhash=wordhash, deptypehash=deptypehash, entityhash=entityhash, relationhash=relationhash, entities=entities, relations=relations, size=#words.idx, ids=ids, get_relative_distance=get_relative_distance}

      
   elseif corpus=="ppi" or corpus=="LLL" or corpus=="AIMed" or corpus=="HPRD50" or corpus=="IEPA" or corpus=="BioInfer" then
      local path, pathdata
      local tot = "train-test"
      if params.mobius then
	 local home = "/home/runuser"
	    path = home .. "/corpus/ppi-eval-standard/"
      else
      	 path = "/home/joel/Bureau/loria/corpus/ppi-eval-standard/"
      end
      pathdata = path .. "extracted/" ..  (copus=="train" and "Train" or "Test")

      local anon = false
      -- wordhash = wordhash or _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/word.txt', params.nword)
      -- entityhash = entityhash or _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/entities.txt', nil)
      relationhash = _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/relations.txt', nil)
      
      if params.arch==3 and params.dtfsz>0 then
	 error("to do")
	 -- 	 deptypehash = _loadhash(path .. '/deptype.txt', nil)
      end
      
      -- poshash = poshash or _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/pos.txt', nil)
      --local ids = loadindices(pathdata .. "raw/" .. corpus .. "sentenceids.txt", params.maxload)
      
      local words  = loadwords(path .. "extracted/" .. tot .. "/" .. corpus .. "/sentences." .. parser .. "_tree_sentences" , wordhash, params.addraw, wordfeature, params.maxload)
      local entities = loadentities(path .. "extracted/" .. tot .. "/" .. corpus .. "/entities.txt", words.sent, words.idx, entityhash, params.maxload, params.wsz)
      pad(words, (params.wsz-1)/2, wordhash.PADDING)
      local relations = loadrelations(path .. "extracted/" .. tot .. "/" .. corpus .. "/relations.txt", relationhash, params.maxload, params.wsz)
      local trees = loadtrees(path .. "extracted/" .. tot .. "/" .. corpus .. "/sentences." .. parser .. "_tree_comp_reps", params.maxload)
      local trees2
      if params.arch==4 or params.arch==5 then
	 trees2 = tree2tree(trees)
      end
      local deptypes
      if (params.arch==3 or params.arch==4 or params.arch==5) and params.dtfsz>0 then
	 deptypes = loadwords(path .. "extracted/" .. tot .. "/" .. corpus .. "/sentences." .. parser .. "_tree_deptype", deptypehash, nil, nil, params.maxload)
	 idx(deptypes)
      end
      local pos
      if params.pfsz~=0 then
	 pos = loadwords(path .. "extracted/" .. tot .. "/" .. corpus .. "/sentences." .. parser .. "_tree_pos" , poshash, params.addraw, nil, params.maxload)
	 pad(pos, (params.wsz-1)/2, poshash.PADDING)
      end

      local select_data
      if trainortest=="train" and params.select_data and params.select_data[corpus] then
	 local f=torch.DiskFile(params.select_data[corpus])
	 select_data = f:readObject()
	 f:close()
	 assert(#words.idx==#select_data)
      end

      
      return {select_data=select_data, corpus=corpus, trees2=trees2, pos=pos, deptypes=deptypes, trees=trees, words=words, poshash=poshash, wordhash=wordhash, deptypehash=deptypehash, entityhash=entityhash, relationhash=relationhash, entities=entities, relations=relations, size=#words.idx, ids=ids, get_relative_distance=get_relative_distance}
      
   elseif corpus=="ADE" then
      local path, pathdata
      local tot = ""--trainortest=="test" and "test" or "train"
      if params.mobius then
	 local home = "/home/runuser"
	 path = home .. "/corpus/ADE/"
      else
	 path = "/home/joel/Bureau/loria/corpus/ADE/"
      end
      pathdata = path .. "extracted/" --..  (copus=="train" and "Train" or "Test")
   
      local anon = false
      -- wordhash = wordhash or _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/word.txt', params.nword)
      -- entityhash = entityhash or _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/entities.txt', nil)
      relationhash = _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/relations.txt', nil)

     
      -- poshash = poshash or _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/pos.txt', nil)
      --local ids = loadindices(pathdata .. "raw/sentenceids.txt", params.maxload)
      
      local words  = loadwords(path .. "extracted/" .. tot .. "/sentences." .. parser .. "_tree_sentences" , wordhash, params.addraw, wordfeature, params.maxload)
      local entities = loadentities(path .. "extracted/" .. tot .. "/entities.txt", words.sent, words.idx, entityhash, params.maxload, params.wsz)
      pad(words, (params.wsz-1)/2, wordhash.PADDING)
      local relations = loadrelations(path .. "extracted/" .. tot .. "/relations.txt", relationhash, params.maxload, params.wsz)
      local trees = loadtrees(path .. "extracted/" .. tot .. "/sentences." .. parser .. "_tree_comp_reps", params.maxload)
      local trees2
      if params.arch==4 or params.arch==5 then
	 trees2 = tree2tree(trees)
      end
      local deptypes
      if (params.arch==3 or params.arch==4 or params.arch==5) and params.dtfsz>0 then
	 deptypes = loadwords(path .. "extracted/" .. tot .. "/sentences." .. parser .. "_tree_deptype", deptypehash, nil, nil, params.maxload)
	 idx(deptypes)
	 -- 	 deptypehash = _loadhash(path .. '/deptype.txt', nil)
      end
      local pos
      if params.pfsz~=0 then
	 pos = loadwords(path .. "extracted/" .. tot .. "/sentences." .. parser .. "_tree_pos" , poshash, params.addraw, nil, params.maxload)
	 pad(pos, (params.wsz-1)/2, poshash.PADDING)
      end

      local select_data
      if trainortest=="train" and params.select_data and params.select_data[corpus] then
	 local f=torch.DiskFile(params.select_data[corpus])
	 select_data = f:readObject()
	 f:close()
	 assert(#words.idx==#select_data)
      end

      
      return {select_data=select_data, corpus=corpus, trees2=trees2, pos=pos, deptypes=deptypes, trees=trees, words=words, poshash=poshash, wordhash=wordhash, deptypehash=deptypehash, entityhash=entityhash, relationhash=relationhash, entities=entities, relations=relations, size=#words.idx, ids=ids, get_relative_distance=get_relative_distance}
   
   elseif corpus=="EUADR_drug_disease" or corpus=="EUADR_drug_target" or corpus=="EUADR_target_disease" then
      local path, pathdata
      local tot = trainortest=="test" and "test" or "train"
      if params.mobius then
	 local home = "/home/runuser"
	    path = home .. "/corpus/EUADR/"
      else
      	 path = "/home/joel/Bureau/loria/corpus/EUADR/"
      end
      pathdata = path .. "extracted/" ..  (copus=="train" and "Train" or "Test")

      local anon = false
      -- wordhash = wordhash or _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/word.txt', params.nword)
      -- entityhash = entityhash or _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/entities.txt', nil)
      relationhash = _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/relations.txt', nil)
      local PA = relationhash["PA"]
      relationhash["NA"] = PA
      relationhash["SA"] = PA
      
      -- poshash = poshash or _loadhash(path .. 'hash' .. (anon and "_anon" or "") .. '/pos.txt', nil)
      --local ids = loadindices(pathdata .. "raw/" .. corpus .. "sentenceids.txt", params.maxload)
      
      local words  = loadwords(path .. "extracted/" .. corpus .. "/sentences." .. parser .. "_tree_sentences" , wordhash, params.addraw, wordfeature, params.maxload)
      local entities = loadentities(path .. "extracted/" .. corpus .. "/entities.txt", words.sent, words.idx, entityhash, params.maxload, params.wsz)
      pad(words, (params.wsz-1)/2, wordhash.PADDING)
      local relations = loadrelations(path .. "extracted/" .. corpus .. "/relations.txt", relationhash, params.maxload, params.wsz)
      local trees = loadtrees(path .. "extracted/" .. corpus .. "/sentences." .. parser .. "_tree_comp_reps", params.maxload)
      local trees2
      if params.arch==4 or params.arch==5 then
	 trees2 = tree2tree(trees)
      end
      local deptypes
      if (params.arch==3 or params.arch==4 or params.arch==5) and params.dtfsz>0 then
	 deptypes = loadwords(path .. "extracted/" .. corpus .. "/sentences." .. parser .. "_tree_deptype", deptypehash, nil, nil, params.maxload)
	 idx(deptypes)
	 -- 	 deptypehash = _loadhash(path .. '/deptype.txt', nil)
      end

      local pos
      if params.pfsz~=0 then
	 pos = loadwords(path .. "extracted/" .. corpus .. "/sentences." .. parser .. "_tree_pos" , poshash, params.addraw, nil, params.maxload)
	 pad(pos, (params.wsz-1)/2, poshash.PADDING)
      end


      local select_data
      if trainortest=="train" and params.select_data and params.select_data[corpus] then
	 local f=torch.DiskFile(params.select_data[corpus])
	 select_data = f:readObject()
	 f:close()
	 assert(#words.idx==#select_data)
      end
      
      return {select_data=select_data, corpus=corpus, trees2=trees2, pos=pos, deptypes=deptypes, trees=trees, words=words, poshash=poshash, wordhash=wordhash, deptypehash=deptypehash, entityhash=entityhash, relationhash=relationhash, entities=entities, relations=relations, size=#words.idx, ids=ids, get_relative_distance=get_relative_distance}
   else
      print(corpus)
      error("unknown corpus")
   end
   
end


function extract_data(data, percentage, sector, remove)

   remove = remove or false
   print("Extracting data from " .. data.corpus .. " remove=" .. (remove and "true" or "false"))
   local size = data.size

   local subsize = math.floor((size*percentage)/100)

   local start = (subsize * (sector-1))+1

   print("\tsize: " .. size .. " subcorpus size: " .. subsize .. " subcorpus start at " .. start)

   local tabs = {words=true, pos=true, deptypes=true}

   local new_size_expected = size - subsize
   
   local newdata = {}
   for k,v in pairs(tabs) do
      if data[k] then
	 local newtab = {}
	 for i=1,subsize do
	    table.insert(newtab, data[k].idx[remove and start or (start+i-1)])
	    if remove then table.remove(data[k].idx, start) end
	    if remove then table.remove(data[k].sent, start) end
	 end
	 newdata[k] = {idx=newtab}
	 setmetatable(newdata[k], getmetatable(data[k]))
      end
   end

   local tabs = {trees2=true,trees=true,entities=true,relations=true,ids=true, select_data=true}
   --local tabs = {words=true}
   
   -- if false then
   --    print("data")
   --    for i=1, 85 do
   -- 	 io.write(#data.entities[i] .. " ")
   --    end
   --    io.write("\n")
   -- end

   for k,v in pairs(tabs) do
      if data[k] then
	 local newtab = {}
	 for i=1,subsize do
	    table.insert(newtab, data[k][remove and start or (start+i-1)])
	    if remove then table.remove(data[k], start) end
	 end
	 newdata[k] = newtab
      end
   end
   
   newdata.entities.nent = data.entities.nent
   newdata.entities.typeent = data.entities.typeent
   newdata.entities.getent = data.entities.getent
   newdata.entities.getenttags = data.entities.getenttags

   newdata.relations.isrelated = data.relations.isrelated
  
   data.size = #data.words.idx
   newdata.size = #newdata.words.idx

   if remove then assert(data.size==new_size_expected, size .. " " .. data.size .. " " .. new_size_expected) end
   if remove then assert(newdata.size==subsize) end
   
   --print("====")
   for k,v in pairs(data) do
      if not newdata[k] then newdata[k] = data[k] end
   end
   
   -- if false then
   --    print("newdata")
   --    for i=1, 35 do
   -- 	 io.write("  ")
   --    end
   --    io.write(" ")
   --    for i=1, 35 do
   -- 	 io.write(#newdata.entities[i] .. " ")
   --    end
   --    io.write("\n")
      
   --    print("olddata")
   --    for i=1, 35 do
   -- 	 io.write(#data.entities[i] .. " ")
   --    end
   --    for i=1, 35 do
   -- 	 io.write("  ")
   --    end
      
   --    for i=36, 50 do
   -- 	 io.write(#data.entities[i] .. " ")
   --    end
   --    io.write("\n")
   -- end

   return newdata
   
end
