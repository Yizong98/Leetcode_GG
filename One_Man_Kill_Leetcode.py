class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        for i in range(len(nums)):
            for j in range(1,len(nums)):
                if (nums[i]+nums[j]== target and i != j):
                    return [i,j]

# Accepted Answer
class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        nums_dict = {}
        for i in range(len(nums)):
            nums_dict[nums[i]] = i
        
        for i in range(len(nums)):
            difference = target - nums[i]
            if (difference in nums_dict and i != nums_dict[difference]):
                return [i,nums_dict[difference]]

# Accepted Answer
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        num1_stack = []
        num2_stack = []
        while (l1.next != None) :
            num1_stack.append(l1.val)
            l1 = l1.next
        num1_stack.append(l1.val)
        while (l2.next != None):
            num2_stack.append(l2.val)
            l2 = l2.next
        num2_stack.append(l2.val)
        num1 = ""
        num2 = ""
        while (len(num1_stack) != 0):
            num1 += str(num1_stack.pop())
        while (len(num2_stack) != 0):
            num2 += str(num2_stack.pop())
        result = str(int(num1) + int(num2))
        output = ListNode(result[-1])
        result = result[:-1]
        current = output
        while(len(result) != 0):
                result, current.next = result[:-1], ListNode(result[-1])
                current = current.next
        return output

#Accepted Answer
class Solution:
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        min_idx = 0
        max_idx = len(nums1) if len(nums1) < len(nums2) else len(nums2)
        length_1 = len(nums1)
        length_2 = len(nums2)
        smaller = nums1 if len(nums1) < len(nums2) else nums2
        larger = nums2 if len(nums1) < len(nums2) else nums1
        while(min_idx <= max_idx):
            i = (min_idx+max_idx)//2
            j = (length_1+length_2+1)//2 - i
            if (i>0  and j < len(larger) and smaller[i-1] > larger[j]):
                max_idx = i - 1
            elif (j>0 and len(smaller)>i and smaller[i] < larger[j-1]):
                min_idx = i + 1
            else:
                if (i==0):
                    median = larger[j-1]
                elif(j==0):
                    median = smaller[i-1]
                else:
                    median = max([smaller[i-1],larger[j-1]])
                break
        if ((length_1 + length_2)%2!=0):
            return median
        if (i == len(smaller)):
            return (median + larger[j])/2
        if (j == len(larger)):
            return (median + smaller[i])/2
        return (median + min([smaller[i],larger[j]]))/2
        
#Accepted Answer
class Solution:
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        if (x < 0):
            return False
        num = []
        elem = x
        while(elem > 0):
            element = elem%10 
            num.append(element)
            elem = elem//10
        track = len(num)
        result = 0
        for i in range(len(num)):
            result += num[i]*10**(track-1)
            track -= 1
        if (x == result):
            return True
        return False
#Accepted Answer
class Solution:
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        negative = True if x < 0 else False
        number_str = ""
        mock_str = str(x)[1:] if negative else str(x)
        while (len(mock_str) > 0):
            number_str += mock_str[-1]
            mock_str = mock_str[:-1]
        final_value = -int(number_str) if negative else int(number_str)
        if (final_value < -2**31 or final_value > 2**31-1):
            return 0
        return final_value

#Accepted Answer
class Solution:
    def licenseKeyFormatting(self, S, K):
        """
        :type S: str
        :type K: int
        :rtype: str
        """
        separate = S.split("-")
        s_with_no_dash = ""
        for char in separate:
            s_with_no_dash += char
        new_str = ""
        while(len(s_with_no_dash) >= K):
            new_str = s_with_no_dash[-K:] + new_str
            s_with_no_dash = s_with_no_dash[:len(s_with_no_dash)-K]
            if (len(s_with_no_dash) > 0):
                new_str = "-" + new_str
        new_str = s_with_no_dash + new_str
        new_str = new_str.upper()
        return new_str
#Accepted Answer
class Solution:
    def nextClosestTime(self, time):
        """
        :type time: str
        :rtype: str
        """
        time_split = time.split(":")
        whole_str = time_split[0] + time_split[1]
        lower_limit_min = time_split[1]
        hr_list = [int(i+j) for i in whole_str for j in whole_str if int(i+j) < 24]
        min_list =  [int(i+j) for i in whole_str for j in whole_str if int(i+j) < 60]
        bigger_hr = sorted(list(set(([i for i in hr_list if i >= int(time_split[0])]))))
        smaller_hr = sorted(list(set([i for i in hr_list if i < int(time_split[0])])))
        bigger_min = sorted(list(set([i for i in min_list if i > int(time_split[1])])))
        smaller_min = sorted(list(set([i for i in min_list if i <= int(time_split[1])])))
        return_val = ""
        print(bigger_hr)
        if bigger_hr:
            if bigger_hr[0] == int(time_split[0]):
                if bigger_min:
                    return_val = str(bigger_hr[0]) + ":" + str(bigger_min[0])
                elif len(bigger_hr) > 1:
                    return_val = str(bigger_hr[1])+ ":" + str(min(min_list))
                else:
                    return_val = str(min(hr_list)) + ":" + str(min(min_list))
            print(return_val)
        else:
            return_val = str(min(hr_list)) + ":" + str(min(min_list))
        if return_val[1] == ":":
            return_val = "0" + return_val
        if return_val[-2] == ":":
            return_val = return_val[:-1] + "0" + return_val[-1]
        return return_val
#Accepted Answer
class Solution:
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if (len(prices) in [0,1]):
            return 0
        max_profit = 0
        min_idx = 0
        smallest = prices[0]
        
        for i in range(1,len(prices)):
            if (prices[i] - prices[min_idx]> max_profit):
                max_profit = prices[i] - prices[min_idx]
            elif(prices[i] - prices[min_idx]<= 0):
                min_idx = i
        return max_profit
#Accepted Answer
class Solution:
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        maxProfit = 0;
        for i in range(1,len(prices)): 
            if (prices[i] - prices[i - 1] > 0):
                maxProfit += prices[i] - prices[i - 1]  
        return maxProfit;
class Solution(object):
    def nextClosestTime(self, time):
        cur = 60 * int(time[:2]) + int(time[3:])
        allowed = {int(x) for x in time if x != ':'}
        while True:
            cur = (cur + 1) % (24 * 60)
            if all(digit in allowed
                    for block in divmod(cur, 60)
                    for digit in divmod(block, 10)):
                return "{:02d}:{:02d}".format(*divmod(cur, 60))
#Accepted Answer
class Solution:
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        if n < 0:
            x = 1/x
            n = -n
        if n == 0:
            return 1
        half = self.myPow(x,n//2)
        if (n % 2 == 0):
            return half * half
        else:
            return half*half*x