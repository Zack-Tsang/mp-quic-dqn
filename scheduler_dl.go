package quic

import (
	"bitbucket.com/marcmolla/gorl"
	"errors"
	"github.com/lucas-clemente/quic-go/internal/protocol"
	"github.com/lucas-clemente/quic-go/internal/utils"
	"time"
	"fmt"
)

type PathStats struct {
	pathID        protocol.PathID
	congWindow    protocol.ByteCount
	bytesInFlight protocol.ByteCount
	nPackets      uint64
	nRetrans      uint64
	nLoss         uint64
	sRTT          time.Duration
	sRTTStdDev    time.Duration
	// == nPackets??
	quota uint
	rTO   time.Duration
}

type AgentScheduler interface {
	Create() error
	SelectPath([]PathStats) (protocol.PathID, error)
	OnSent(offset protocol.ByteCount, size protocol.ByteCount, done bool)
	GetQUICThroughput(delta time.Duration) gorl.Output
	CloseSession(goodput float64, id protocol.ConnectionID)
}

type DQNAgentScheduler struct {
	weightsFileName string
	epsilon         float32
	agent           gorl.Agent
	packetHistory   map[protocol.ByteCount]protocol.ByteCount
	previousPacket  time.Time
	previousReward	gorl.Output
	offlineWriter	  OfflineWriter
}

func (d *DQNAgentScheduler) OnSent(offset protocol.ByteCount, size protocol.ByteCount, done bool){
	if _, ok := d.packetHistory[offset]; ok == false{
		d.packetHistory[offset] = size
	}
	if done{
		//d.CloseSession()
	}
}

func (d *DQNAgentScheduler) GetQUICThroughput(delta time.Duration) gorl.Output{
	var goodput gorl.Output
	for _, value := range d.packetHistory{
		goodput += gorl.Output(value)
	}
	//Clear history
	d.packetHistory = make(map[protocol.ByteCount]protocol.ByteCount)
	return goodput*gorl.Output(time.Second.Nanoseconds())/gorl.Output(delta.Nanoseconds())*8
}

func (d *DQNAgentScheduler) Create() error {
	var myPolicy gorl.PolicySelector
	start :=time.Now()

	d.packetHistory = make(map[protocol.ByteCount]protocol.ByteCount)

	if d.epsilon != 0. {
		myPolicy = &gorl.E_greedy{Epsilon: d.epsilon}
	} else {
		myPolicy = &gorl.ArgMax{}
	}
	myModel := gorl.DNN{}
	myModel.AddLayer(&gorl.Dense{Size: 16, ActFunction: gorl.Relu})
	myModel.AddLayer(&gorl.Dense{Size: 16, ActFunction: gorl.Relu})
	myModel.AddLayer(&gorl.Dense{Size: 2, ActFunction: gorl.Linear})

	d.agent = &gorl.DQNAgent{Policy: myPolicy, QModel: &myModel}

	if d.weightsFileName == ""{
		d.weightsFileName = "../data/blank_weights.h5f"
	}

	d.agent.LoadWeights(d.weightsFileName)

	d.offlineWriter = OfflineWriter{}
	d.offlineWriter.Init()

	elapsed := time.Since(start)
	utils.Infof("Agent created in %d us", elapsed.Nanoseconds()/1000)
	return nil
}

func (d *DQNAgentScheduler) SelectPath(stats []PathStats) (protocol.PathID, error) {
	if len(stats) == 1 {
		return stats[0].pathID, nil
	}
	if len(stats) != 2 {
		return protocol.InitialPathID, errors.New("only two paths supported")
	}
	var firstPath, secondPath PathStats
	if stats[0].pathID < stats[1].pathID {
		firstPath, secondPath = stats[0], stats[1]
	} else {
		firstPath, secondPath = stats[1], stats[0]
	}
	fretrans, floss := gorl.Output(0.), gorl.Output(0.)
	if firstPath.nPackets != 0{
		fretrans, floss = gorl.Output(firstPath.nRetrans) / gorl.Output(firstPath.nPackets),
		gorl.Output(firstPath.nLoss) / gorl.Output(firstPath.nPackets)
	}
	sretrans, sloss := gorl.Output(0.), gorl.Output(0.)
	if secondPath.nPackets != 0{
		sretrans, sloss = gorl.Output(secondPath.nRetrans) / gorl.Output(secondPath.nPackets),
			gorl.Output(secondPath.nLoss) / gorl.Output(secondPath.nPackets)
	}

	state := gorl.Vector{
		gorl.Output(float32(firstPath.congWindow - firstPath.bytesInFlight) / float32(firstPath.congWindow)),
		fretrans,
		floss,
		normalizeTimes(firstPath.sRTT),
		normalizeTimes(firstPath.sRTTStdDev),
		normalizeTimes(firstPath.rTO),
		gorl.Output(float32(secondPath.congWindow - secondPath.bytesInFlight) / float32(secondPath.congWindow)),
		sretrans,
		sloss,
		normalizeTimes(secondPath.sRTT),
		normalizeTimes(secondPath.sRTTStdDev),
		normalizeTimes(secondPath.rTO),
	}
	if utils.Debug() {
		utils.Debugf("Input state: %v", state)
	}
	var rewardStr string
	if d.previousPacket.IsZero(){
		//First packet of the session
		d.previousPacket = time.Now()
		rewardStr = "START"
	}else{
		reward := d.GetQUICThroughput(time.Since(d.previousPacket))
		utils.Debugf("goodput: %f, delta: %d", reward,
			time.Since(d.previousPacket).Nanoseconds())

		d.previousPacket = time.Now()
		if reward != 0 {
			reward =(reward - d.previousReward) / reward
			d.previousReward = reward
		}
		rewardStr = fmt.Sprintf("%f", reward)
		if reward != 0{
			utils.Infof("Reward: %f, str: %s", reward, rewardStr)
		}
	}
	outputPath := d.agent.GetAction(state)
	d.saveOffline(state, outputPath, rewardStr)
	if outputPath == 0{
		utils.Debugf("Selected Path %d", firstPath.pathID)
		return firstPath.pathID, nil
	}else if outputPath == 1{
		utils.Debugf("Selected Path %d", secondPath.pathID)
		return secondPath.pathID, nil
	}
	return 0, errors.New("weights not found!!! Please load weight file")
}

func normalizeTimes(stat time.Duration) gorl.Output {
	return gorl.Output(stat.Nanoseconds()) / gorl.Output(time.Second.Nanoseconds())
}

func (d *DQNAgentScheduler)saveOffline(state gorl.Vector, path int, reward string){
	d.offlineWriter.Append([]string{reward, fmt.Sprint(state), fmt.Sprint(path)})
}

func (d *DQNAgentScheduler)CloseSession(goodput float64, id protocol.ConnectionID){
	//Set to 0
	d.previousPacket = time.Time{}

	// RTT 0
	reward := goodput * 10 / (float64(17.71069254) * 2)

	d.offlineWriter.Close(fmt.Sprint(reward), fmt.Sprint(id))
}
