package quic

import (
	"bitbucket.com/marcmolla/gorl"
	"errors"
	"github.com/lucas-clemente/quic-go/internal/protocol"
	"github.com/lucas-clemente/quic-go/internal/utils"
	"time"
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
}

type DQNAgentScheduler struct {
	weightsFileName string
	epsilon         float32
	agent           gorl.Agent
}

func (d *DQNAgentScheduler) Create() error {
	var myPolicy gorl.PolicySelector

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

	return nil
}

func (d *DQNAgentScheduler) SelectPath(stats []PathStats) (protocol.PathID, error) {
	if len(stats) == 1 {
		return stats[0].pathID, nil
	}
	if len(stats) != 2 {
		return protocol.InitialPathID, errors.New("Only two paths supported")
	}
	var firstPath, secondPath PathStats
	if stats[0].pathID < stats[1].pathID {
		firstPath, secondPath = stats[0], stats[1]
	} else {
		firstPath, secondPath = stats[1], stats[0]
	}
	state := gorl.Vector{
		gorl.Output((firstPath.congWindow - firstPath.bytesInFlight) / firstPath.congWindow),
		gorl.Output(firstPath.nRetrans / firstPath.nPackets),
		gorl.Output(firstPath.nLoss / firstPath.nPackets),
		normalizeTimes(firstPath.sRTT),
		normalizeTimes(firstPath.sRTTStdDev),
		normalizeTimes(firstPath.rTO),
		gorl.Output((secondPath.congWindow - secondPath.bytesInFlight) / secondPath.congWindow),
		gorl.Output(secondPath.nRetrans / secondPath.nPackets),
		gorl.Output(secondPath.nLoss / secondPath.nPackets),
		normalizeTimes(secondPath.sRTT),
		normalizeTimes(secondPath.sRTTStdDev),
		normalizeTimes(secondPath.rTO),
	}
	if utils.Debug() {
		utils.Debugf("Input state: %s", state)
	}
	return protocol.PathID(d.agent.GetAction(state)), nil
}

func normalizeTimes(stat time.Duration) gorl.Output {
	return gorl.Output(stat.Nanoseconds() / time.Second.Nanoseconds())
}